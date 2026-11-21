import os
import torch
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import numpy as np

from src.postprocess import PostProcessor
import numpy as np

class MLEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLEngine, cls).__new__(cls)
            cls._instance.setup()
        return cls._instance

    def setup(self):
        model_path = os.getenv("MODEL_PATH", "weights/Golden.pt")
        print(f"Loading YOLO model from {model_path}...")
        self.yolo_model = YOLO(model_path)  # Custom weights
        
        print("Loading CLIP model...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Initializing Post-Processor...")
        self.post_processor = PostProcessor()
        
        print(f"ML Models loaded. Using device: {self.device}")

    def detect_objects(self, image: Image.Image, conf_threshold: float = 0.25, iof_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect objects in the image using YOLO, then apply post-processing.
        Returns a list of dicts with: label, confidence, box (xyxy), crop (PIL Image)
        """
        # 1. Raw Inference
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)
        
        # 2. Extract Raw Data
        raw_boxes = []
        raw_scores = []
        raw_class_ids = []
        
        # YOLO results are a list (one per image), we only process one image here
        res = results[0]
        if res.boxes:
            raw_boxes = res.boxes.xyxy.cpu().numpy().tolist()
            raw_scores = res.boxes.conf.cpu().numpy().tolist()
            raw_class_ids = res.boxes.cls.cpu().numpy().astype(int).tolist()
            
        # 3. Post-Processing
        final_boxes, final_scores, final_ids = self.post_processor.apply_rules(
            raw_boxes, raw_scores, raw_class_ids, iof_threshold=iof_threshold
        )
        
        detections = []
        for box, score, cls_id in zip(final_boxes, final_scores, final_ids):
            # Parse Box
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            
            # Label
            label = self.yolo_model.names[cls_id]
            
            # Crop
            crop = image.crop((x1, y1, x2, y2))
            
            detections.append({
                "label": label,
                "confidence": float(score),
                "box": [x1, y1, x2, y2],
                "crop": crop
            })
        
        return detections

    def get_embedding(self, image: Image.Image) -> List[float]:
        """
        Generate CLIP embedding for an image (or crop).
        """
        embedding = self.clip_model.encode(image)
        return embedding.tolist()

# Global instance
ml_engine = MLEngine()
