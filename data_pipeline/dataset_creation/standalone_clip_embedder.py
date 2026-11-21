#!/usr/bin/env python3
"""
CLIP Embedder - Her klasörden max 5000 resim alıp embedding çıkarır.
Sonuçları parquet dosyasına kaydeder.

Spot instance desteği: Checkpoint/resume sistemi ile kesintilerde kaldığı yerden devam eder.
"""

import os
import json
import argparse
import signal
import sys
import atexit
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional, Set
from datetime import datetime
import random
import requests

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# CLIP için open_clip kullanıyoruz (daha iyi performans)
try:
    import open_clip
    USE_OPEN_CLIP = True
except ImportError:
    from transformers import CLIPProcessor, CLIPModel
    USE_OPEN_CLIP = False


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}
MAX_IMAGES_PER_FOLDER = 5000
CHECKPOINT_FILE = "checkpoint.json"

# Global state for graceful shutdown
_shutdown_requested = False
_current_state = None


def get_image_paths(folder: Path) -> List[Path]:
    """Klasördeki tüm resim dosyalarını bul."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f'*{ext}'))
        images.extend(folder.glob(f'*{ext.upper()}'))
    return images


def signal_handler(signum, frame):
    """Graceful shutdown için signal handler."""
    global _shutdown_requested
    print(f"\n⚠️  Sinyal alındı ({signum}), güvenli kapatma başlatılıyor...")
    _shutdown_requested = True


def setup_signal_handlers():
    """SIGTERM ve SIGINT için handler'ları ayarla."""
    signal.signal(signal.SIGTERM, signal_handler)  # Spot interruption
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C


def check_spot_interruption():
    """EC2 Spot interruption notice kontrolü."""
    try:
        # EC2 Instance Metadata Service v2 (IMDSv2)
        token_response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=1
        )
        token = token_response.text
        
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=1
        )
        if response.status_code == 200:
            return True  # Interruption notice var!
    except:
        pass
    return False


def spot_monitor_thread():
    """Arka planda spot interruption kontrolü."""
    global _shutdown_requested
    while not _shutdown_requested:
        if check_spot_interruption():
            print("\n🔴 SPOT INTERRUPTION ALGILANDI! Güvenli kapatma başlatılıyor...")
            _shutdown_requested = True
            break
        time.sleep(5)  # Her 5 saniyede bir kontrol


def start_spot_monitor():
    """Spot monitor thread'i başlat."""
    monitor = threading.Thread(target=spot_monitor_thread, daemon=True)
    monitor.start()
    print("👀 EC2 Spot interruption monitor başlatıldı")
    

def save_checkpoint(output_dir: Path, state: dict):
    """Checkpoint kaydet."""
    checkpoint_path = output_dir / CHECKPOINT_FILE
    temp_path = output_dir / f"{CHECKPOINT_FILE}.tmp"
    
    state['last_updated'] = datetime.now().isoformat()
    
    # Atomic write
    with open(temp_path, 'w') as f:
        json.dump(state, f, indent=2)
    temp_path.rename(checkpoint_path)
    
    print(f"💾 Checkpoint kaydedildi: {len(state.get('completed_folders', []))} klasör tamamlandı")


def load_checkpoint(output_dir: Path) -> Optional[dict]:
    """Varsa checkpoint'i yükle."""
    checkpoint_path = output_dir / CHECKPOINT_FILE
    
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path) as f:
            state = json.load(f)
        print(f"📂 Checkpoint bulundu: {state.get('last_updated', 'unknown')}")
        print(f"   Tamamlanan klasörler: {len(state.get('completed_folders', []))}")
        return state
    except Exception as e:
        print(f"⚠️  Checkpoint okunamadı: {e}")
        return None


def get_completed_folders(output_dir: Path) -> Set[str]:
    """Zaten tamamlanmış klasörleri bul."""
    checkpoint = load_checkpoint(output_dir)
    if checkpoint:
        return set(checkpoint.get('completed_folders', []))
    return set()


def load_clip_model(device: str, model_name: str = "ViT-B-32"):
    """CLIP modelini yükle."""
    if USE_OPEN_CLIP:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained='laion2b_s34b_b79k'
        )
        model = model.to(device).eval()
        return model, preprocess, None
    else:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device).eval()
        return model, None, processor


def get_embeddings_batch(
    model, 
    preprocess, 
    processor,
    image_paths: List[Path], 
    device: str,
    batch_size: int = 64
) -> Tuple[List[str], np.ndarray]:
    """Batch halinde embedding çıkar."""
    global _shutdown_requested
    
    valid_paths = []
    embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        # Shutdown kontrolü
        if _shutdown_requested:
            print("⚠️  İşlem durduruluyor...")
            break
            
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_valid_paths = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                if USE_OPEN_CLIP:
                    img = preprocess(img)
                batch_images.append(img)
                batch_valid_paths.append(str(path))
            except Exception as e:
                print(f"Resim yüklenemedi: {path} - {e}")
                continue
        
        if not batch_images:
            continue
            
        with torch.no_grad():
            if USE_OPEN_CLIP:
                images_tensor = torch.stack(batch_images).to(device)
                features = model.encode_image(images_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy()
            else:
                inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy()
        
        embeddings.append(features)
        valid_paths.extend(batch_valid_paths)
    
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])
    
    return valid_paths, embeddings


def process_folders(
    root_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 64,
    max_per_folder: int = MAX_IMAGES_PER_FOLDER,
    resume: bool = True
):
    """Tüm klasörleri işle ve embeddings kaydet."""
    global _shutdown_requested, _current_state
    
    # Signal handler'ları ayarla
    setup_signal_handlers()
    
    # EC2 Spot monitor başlat (opsiyonel, hata verirse devam et)
    try:
        start_spot_monitor()
    except Exception as e:
        print(f"ℹ️  Spot monitor başlatılamadı (EC2 dışında olabilirsiniz): {e}")
    
    print(f"Model yükleniyor (device: {device})...")
    model, preprocess, processor = load_clip_model(device)
    
    # Alt klasörleri bul
    folders = [d for d in root_dir.iterdir() if d.is_dir()]
    
    if not folders:
        # Eğer alt klasör yoksa, root'u klasör olarak kullan
        folders = [root_dir]
    
    # Alfabetik sırala (deterministic order)
    folders = sorted(folders, key=lambda x: x.name)
    
    print(f"Bulunan klasör sayısı: {len(folders)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint kontrolü
    completed_folders = set()
    if resume:
        completed_folders = get_completed_folders(output_dir)
        if completed_folders:
            print(f"✅ Daha önce tamamlanan klasörler atlanacak: {len(completed_folders)}")
    
    # State başlat
    _current_state = {
        "completed_folders": list(completed_folders),
        "total_folders": len(folders),
        "root_dir": str(root_dir),
        "started_at": datetime.now().isoformat()
    }
    
    summary = {
        "folders": {},
        "total_images": 0,
        "embedding_dim": None
    }
    
    # Önceki summary'yi yükle
    summary_file = output_dir / "summary.json"
    if summary_file.exists() and resume:
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            print(f"📂 Önceki summary yüklendi: {summary['total_images']} resim")
        except:
            pass
    
    for folder_idx, folder in enumerate(folders):
        folder_name = folder.name
        
        # Shutdown kontrolü
        if _shutdown_requested:
            print("\n⚠️  Shutdown isteği alındı, checkpoint kaydediliyor...")
            save_checkpoint(output_dir, _current_state)
            break
        
        # Zaten tamamlandıysa atla
        if folder_name in completed_folders:
            print(f"⏭️  Atlanıyor (zaten tamamlandı): {folder_name}")
            continue
        
        print(f"\n{'='*50}")
        print(f"📁 [{folder_idx+1}/{len(folders)}] Klasör: {folder_name}")
        
        # Seçilmiş resim listesi var mı kontrol et (partial resume için)
        selection_file = output_dir / f"{folder_name}_selection.json"
        
        if selection_file.exists():
            # Önceden seçilmiş listeyi yükle
            print(f"📂 Önceki seçim yükleniyor: {selection_file}")
            with open(selection_file) as f:
                selection_data = json.load(f)
            image_paths = [Path(p) for p in selection_data['selected_images']]
            print(f"Yüklenen resim sayısı: {len(image_paths)}")
        else:
            # Resimleri bul
            image_paths = get_image_paths(folder)
            print(f"Bulunan resim: {len(image_paths)}")
            
            if len(image_paths) == 0:
                print("Resim bulunamadı, atlanıyor...")
                _current_state["completed_folders"].append(folder_name)
                continue
            
            # Max sayıya göre rastgele seç
            if len(image_paths) > max_per_folder:
                print(f"Rastgele {max_per_folder} resim seçiliyor...")
                random.seed(42 + hash(folder_name) % 10000)  # Klasör bazlı seed
                image_paths = random.sample(image_paths, max_per_folder)
            
            # Seçimi kaydet (partial resume için)
            selection_data = {
                'folder': folder_name,
                'total_found': len(get_image_paths(folder)),
                'selected_count': len(image_paths),
                'selected_images': [str(p) for p in image_paths],
                'created_at': datetime.now().isoformat()
            }
            with open(selection_file, 'w') as f:
                json.dump(selection_data, f)
            print(f"💾 Seçim kaydedildi: {selection_file}")
        
        # Embeddings çıkar
        paths, embeddings = get_embeddings_batch(
            model, preprocess, processor,
            image_paths, device, batch_size
        )
        
        # Shutdown sırasında yarıda kaldıysa kaydetme
        if _shutdown_requested:
            print("⚠️  Klasör yarıda kesildi, kaydedilmiyor...")
            save_checkpoint(output_dir, _current_state)
            break
        
        if len(paths) == 0:
            print("Geçerli resim bulunamadı, atlanıyor...")
            _current_state["completed_folders"].append(folder_name)
            continue
        
        # DataFrame oluştur
        df = pd.DataFrame({
            'image_path': paths,
            'folder': folder_name
        })
        
        # Embeddings'leri ayrı numpy array olarak kaydet
        embeddings_file = output_dir / f"{folder_name}_embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        # Metadata'yı parquet olarak kaydet
        metadata_file = output_dir / f"{folder_name}_metadata.parquet"
        df.to_parquet(metadata_file, index=False)
        
        print(f"✅ Kaydedildi: {len(paths)} resim")
        print(f"  - Embeddings: {embeddings_file}")
        print(f"  - Metadata: {metadata_file}")
        
        summary["folders"][folder_name] = {
            "count": len(paths),
            "embeddings_file": str(embeddings_file),
            "metadata_file": str(metadata_file)
        }
        summary["total_images"] += len(paths)
        summary["embedding_dim"] = embeddings.shape[1] if len(embeddings.shape) > 1 else None
        
        # Klasör tamamlandı olarak işaretle
        _current_state["completed_folders"].append(folder_name)
        
        # Selection dosyasını temizle (artık gerekli değil)
        if selection_file.exists():
            selection_file.unlink()
        
        # Her klasör sonunda checkpoint kaydet
        save_checkpoint(output_dir, _current_state)
        
        # Summary'yi de güncelle
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Final summary
    print(f"\n{'='*50}")
    if _shutdown_requested:
        print(f"⚠️  İŞLEM DURDURULDU (checkpoint kaydedildi)")
        print(f"   Devam etmek için aynı komutu tekrar çalıştırın.")
    else:
        print(f"✅ TAMAMLANDI!")
        # Checkpoint dosyasını sil (başarıyla tamamlandı)
        checkpoint_path = output_dir / CHECKPOINT_FILE
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("   Checkpoint temizlendi.")
    
    print(f"Toplam işlenen resim: {summary['total_images']}")
    print(f"Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='CLIP Embedder - Resimlerden embedding çıkar')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Resimlerin bulunduğu ana klasör')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Embeddings çıktı klasörü')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Batch boyutu (default: 64)')
    parser.add_argument('--max-per-folder', '-m', type=int, default=MAX_IMAGES_PER_FOLDER,
                        help=f'Klasör başına max resim sayısı (default: {MAX_IMAGES_PER_FOLDER})')
    parser.add_argument('--device', '-d', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Checkpoint kullanma, sıfırdan başla')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"🚀 CLIP Embedder başlatılıyor...")
    print(f"   Device: {device}")
    print(f"   Resume: {'Hayır' if args.no_resume else 'Evet'}")
    
    process_folders(
        root_dir=Path(args.input),
        output_dir=Path(args.output),
        device=device,
        batch_size=args.batch_size,
        max_per_folder=args.max_per_folder,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
