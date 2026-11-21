import os
import json
import threading
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from queue import Queue
import gc

# --- Ultralytics (YOLO) ---
try:
    from ultralytics import YOLOWorld
    print("✅ Ultralytics (YOLO-World) yüklendi.")
except ImportError:
    raise ImportError("❌ 'ultralytics' bulunamadı. Lütfen 'pip install ultralytics' çalıştırın.")

def json_safe(obj):
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.item()
    return str(obj)

class FurnitureAnnotatorYOLO:
    def __init__(self, golden_dataset_path, images_root_path, output_dir="yolo_annotations", conf_threshold=0.10):
        
        self.images_root = Path(images_root_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.golden_df = pd.read_parquet(golden_dataset_path)
        self.conf_threshold = conf_threshold
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Cihaz: {self.device} (YOLO-World v2 Large kullanılıyor)")
        
        # --- MODEL YÜKLEME ---
        # 'yolov8l-worldv2.pt' (Large) en iyi doğruluk/hız dengesini verir.
        # Daha da hız istersen 'yolov8s-worldv2.pt' kullanabilirsin.
        self.model = YOLOWorld('yolov8l-worldv2.pt') 
        
        # Kategorileri Hazırla
        self.categories = self._generate_categories()
        self.category_names = [c["name"] for c in self.categories]
        
        # YOLO-World'e sınıfları TEK SEFERDE tanıtıyoruz (Offline Vocabulary)
        print("🧠 Kategoriler YOLO-World'e gömülüyor (Offline Mode)...")
        self.model.set_classes(self.category_names)
        print(f"✅ {len(self.category_names)} kategori tanımlandı.")

    def _generate_categories(self):
        # 114 Sınıflı Hiyerarşi
        hierarchy = {
            "Furniture": ["sofa", "couch", "armchair", "loveseat", "chair", "dining chair", "office chair", "stool", "bench", "ottoman", "pouf", "dining table", "coffee table", "side table", "nightstand", "bedside table", "desk", "computer desk", "console table", "wardrobe", "closet", "cabinet", "cupboard", "chest of drawers", "dresser", "bookshelf", "bookcase", "TV stand", "media unit", "bed frame", "headboard", "coat rack", "shoe rack"],
            "Textile": ["rug", "carpet", "floor mat", "runner", "doormat", "curtain", "drape", "window blind", "pillow", "cushion", "throw pillow", "blanket", "throw blanket", "duvet", "quilt", "comforter", "bed spread", "messy bed sheets", "bedding", "crumpled blanket", "towel", "bath mat"],
            "Lighting": ["floor lamp", "standing lamp", "reading lamp", "table lamp", "desk lamp", "bedside lamp", "ceiling light", "pendant light", "chandelier", "wall sconce", "wall lamp", "lampshade"],
            "Decoration": ["vase", "flower pot", "planter", "potted plant", "indoor plant", "mirror", "wall mirror", "standing mirror", "picture frame", "photo frame", "painting", "wall art", "poster", "sculpture", "statue", "figurine", "bust", "candle", "candlestick", "candle holder", "decorative bowl", "tray", "basket", "storage box", "wall clock", "table clock"],
            "Appliance": ["television", "tv screen", "monitor", "display", "refrigerator", "fridge", "washing machine", "dryer", "dishwasher", "oven", "stove", "microwave", "toaster", "kettle", "coffee maker", "speaker", "soundbar", "air conditioner", "fan", "heater", "radiator"]
        }
        cats = []
        idx = 1
        for supercat, items in hierarchy.items():
            for item in items:
                # name: leaf (sofa), supercategory: parent (Furniture)
                cats.append({"id": idx, "name": item, "supercategory": supercat})
                idx += 1
        return cats

    def process_image(self, img_path):
        """YOLO-World Inference"""
        # YOLO path string kabul eder, PIL açmaya gerek yok (Ultralytics kendi halleder)
        # persist=False: Takip modu kapalı
        # verbose=False: Konsolu kirletme
        results = self.model.predict(
            str(img_path), 
            conf=self.conf_threshold, 
            iou=0.5, # NMS threshold
            verbose=False,
            device=self.device
        )
        
        final_annotations = []
        result = results[0] # Tek resim işliyoruz
        
        if len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                # Class ID (Bizim listemizdeki sıraya göre)
                cls_idx = int(box.cls)
                score = float(box.conf)
                
                # Bizim kategori ID'mizi bul (Liste sırası + 1 mi yoksa ID mi?)
                # categories listesi sıralı oluşturulduğu için:
                # model.names[cls_idx] -> class name
                # Bizim category['id'] -> cls_idx + 1 (Çünkü id 1'den başlattık)
                user_cat_id = self.categories[cls_idx]["id"]
                
                # Koordinatlar: xyxy -> xywh (COCO formatı için: x_min, y_min, width, height)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                bbox = [x1, y1, w, h]
                area = w * h
                
                # YOLO Maske vermez, ama format bozulmasın diye sahte segmentation ekleyebiliriz
                # ya da boş bırakabiliriz. Burada Polygon Box veriyoruz.
                poly = [[x1, y1, x2, y1, x2, y2, x1, y2]] 
                
                final_annotations.append({
                    "id": i + 1,
                    "category_id": user_cat_id,
                    "segmentation": [poly], # RLE yerine Poligon Box (Hız için)
                    "bbox": bbox,
                    "score": score,
                    "area": area,
                    "iscrowd": 0
                })
                
        return final_annotations

    def run_pipeline(self, limit=None):
        df_to_process = self.golden_df
        if limit:
            df_to_process = self.golden_df.head(limit)
            print(f"⚠️ TEST MODU: {limit} resim.")
        else:
            print(f"🚀 TAM MOD: {len(self.golden_df)} resim.")

        task_queue = Queue(maxsize=50) # YOLO çok hızlı, kuyruğu büyüttük
        save_queue = Queue()
        
        # Producer
        def producer():
            print("🧵 Yükleyici başladı...")
            for _, row in df_to_process.iterrows():
                folder = row['folder_name']
                img_name = row['image_name']
                img_path = self.images_root / folder / img_name
                out_path = self.output_dir / f"{folder}_{Path(img_name).stem}.json"
                
                if not out_path.exists() and img_path.exists():
                    task_queue.put((img_path, row, out_path))
            task_queue.put(None)

        # Saver
        def saver():
            print("💾 Kaydedici başladı...")
            while True:
                item = save_queue.get()
                if item is None: break
                out_path, data = item
                with open(out_path, 'w') as f:
                    json.dump(data, f, indent=2, default=json_safe)
                save_queue.task_done()

        t_prod = threading.Thread(target=producer)
        t_save = threading.Thread(target=saver)
        t_prod.start()
        t_save.start()

        # Main Loop (YOLO Inference)
        pbar = tqdm(total=len(df_to_process), desc="YOLO Inference")
        
        while True:
            if task_queue.empty() and not t_prod.is_alive(): break
            
            item = task_queue.get()
            if item is None: break
            
            img_path, meta_row, out_path = item
            
            try:
                # YOLO Inference (Direkt path veriyoruz, yükleme ile zaman kaybetmiyoruz)
                anns = self.process_image(img_path)
                
                output_data = {
                    "image_meta": meta_row.to_dict(),
                    "annotations": anns,
                    "categories": self.categories
                }
                save_queue.put((out_path, output_data))
                
            except Exception as e:
                print(f"Hata: {e}")
            
            pbar.update(1)
            task_queue.task_done()

        save_queue.put(None)
        t_prod.join()
        t_save.join()
        pbar.close()
        print(f"✅ Bitti. Çıktılar: {self.output_dir}")

if __name__ == "__main__":
    annotator = FurnitureAnnotatorYOLO(
        golden_dataset_path="../golden_dataset.parquet",
        images_root_path="../images",
        output_dir="../annotations_yolo",
        conf_threshold=0.15 # Biraz daha toleranslı
    )
    
    # Test için
    annotator.run_pipeline(limit=None)