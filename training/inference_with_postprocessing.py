"""
Inference and Post-Processing Pipeline

This script performs model inference with advanced post-processing rules to improve
detection quality. Post-processing handles overlapping detections and conflicting
class predictions.

Post-Processing Rules:
1. Conflict Detection: Different classes with IoU > 0.97 are removed
2. Class Merging: Same classes detected at same location are merged
3. Confidence Filtering: Removes low-confidence detections
4. IoU Suppression: Removes highly overlapping detections

Output:
- Annotated images with bounding boxes
- JSON results with detailed predictions
- Processing logs with filtering information
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime


class DetectionPostProcessor:
    """
    Post-processes YOLO detections to improve quality and resolve conflicts
    """
    
    def __init__(self, class_names, iof_threshold=0.85):
        """
        Initialize post-processor
        
        Args:
            class_names: Dictionary of class_id: class_name
            iof_threshold: Intersection over Foreground threshold for box relationships
        """
        self.class_names = class_names
        self.iof_threshold = iof_threshold
        self.processing_log = []
    
    def calculate_iof(self, box1, box2):
        """
        Calculate Intersection over Foreground (IoF)
        Measures overlap between two boxes relative to smaller box
        
        Args:
            box1, box2: Bounding boxes [x1, y1, x2, y2]
        
        Returns:
            IoF value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection coordinates
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        if box1_area == 0:
            return 0
        return intersection_area / box1_area
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU)
        Standard metric for bounding box overlap
        
        Args:
            box1, box2: Bounding boxes [x1, y1, x2, y2]
        
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Box areas
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union
        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def is_center_inside(self, inner_box, outer_box):
        """
        Check if center of inner box is inside outer box
        
        Args:
            inner_box, outer_box: Bounding boxes [x1, y1, x2, y2]
        
        Returns:
            True if center is inside, False otherwise
        """
        cx = (inner_box[0] + inner_box[2]) / 2
        cy = (inner_box[1] + inner_box[3]) / 2
        x1, y1, x2, y2 = outer_box
        return (x1 < cx < x2) and (y1 < cy < y2)
    
    def apply_postprocessing_rules(self, boxes, scores, class_ids, image_name="unknown"):
        """
        Apply post-processing rules to filter and merge detections
        
        Args:
            boxes: List of bounding boxes
            scores: List of confidence scores
            class_ids: List of class IDs
            image_name: Image identifier for logging
        
        Returns:
            Filtered boxes, scores, and class_ids
        """
        n_boxes = len(boxes)
        if n_boxes == 0:
            return [], [], []
        
        keep = [True] * n_boxes
        
        # Rule 1: Remove conflicting classes (high IoU with different classes)
        for i in range(n_boxes):
            if not keep[i]:
                continue
            for j in range(i + 1, n_boxes):
                if not keep[j]:
                    continue
                
                if class_ids[i] != class_ids[j]:
                    iou = self.calculate_iou(boxes[i], boxes[j])
                    
                    if iou > 0.97:
                        keep[i] = False
                        keep[j] = False
                        
                        log_entry = {
                            "image": image_name,
                            "action": "conflict_removal",
                            "class1": self.class_names[class_ids[i]],
                            "class2": self.class_names[class_ids[j]],
                            "iou": iou
                        }
                        self.processing_log.append(log_entry)
        
        # Rule 2: Merge same classes with high overlap
        for i in range(n_boxes):
            if not keep[i]:
                continue
            for j in range(i + 1, n_boxes):
                if not keep[j]:
                    continue
                
                if class_ids[i] == class_ids[j]:
                    iof_ij = self.calculate_iof(boxes[j], boxes[i])
                    iof_ji = self.calculate_iof(boxes[i], boxes[j])
                    
                    if iof_ij > self.iof_threshold or iof_ji > self.iof_threshold:
                        # Merge: keep higher confidence
                        if scores[i] >= scores[j]:
                            keep[j] = False
                        else:
                            keep[i] = False
                        
                        log_entry = {
                            "image": image_name,
                            "action": "class_merge",
                            "class": self.class_names[class_ids[i]],
                            "iof": max(iof_ij, iof_ji)
                        }
                        self.processing_log.append(log_entry)
        
        # Apply filtering
        filtered_boxes = [boxes[i] for i in range(n_boxes) if keep[i]]
        filtered_scores = [scores[i] for i in range(n_boxes) if keep[i]]
        filtered_classes = [class_ids[i] for i in range(n_boxes) if keep[i]]
        
        return filtered_boxes, filtered_scores, filtered_classes


def run_inference_with_postprocessing(
    model_path,
    image_source,
    output_dir,
    conf_threshold=0.35,
    iou_threshold=0.45,
    device=0
):
    """
    Run inference on images with post-processing
    
    Args:
        model_path: Path to trained model weights
        image_source: Image file or directory
        output_dir: Directory for results
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device: GPU device ID or 'cpu'
    """
    
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(image_source):
        raise FileNotFoundError(f"Image source not found: {image_source}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize
    model = YOLO(model_path)
    post_processor = DetectionPostProcessor(
        class_names=model.names,
        iof_threshold=0.85
    )
    
    # Get image list
    image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if os.path.isdir(image_source):
        image_list = []
        for ext in image_extensions:
            image_list.extend(Path(image_source).rglob(f"*{ext}"))
    else:
        image_list = [Path(image_source)]
    
    print(f"Processing {len(image_list)} images...")
    
    results_data = []
    
    for image_path in tqdm(image_list, desc="Inference"):
        # Run YOLO inference
        results = model.predict(
            str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )
        
        if not results or len(results) == 0:
            continue
        
        result = results[0]
        
        # Extract detections
        boxes = []
        scores = []
        class_ids = []
        
        if result.boxes is not None:
            for box in result.boxes:
                boxes.append(box.xyxy[0].cpu().numpy().tolist())
                scores.append(float(box.conf))
                class_ids.append(int(box.cls))
        
        # Apply post-processing
        filtered_boxes, filtered_scores, filtered_classes = (
            post_processor.apply_postprocessing_rules(
                boxes, scores, class_ids, image_path.name
            )
        )
        
        # Save annotated image
        image = cv2.imread(str(image_path))
        for box, score, cls_id in zip(filtered_boxes, filtered_scores, filtered_classes):
            x1, y1, x2, y2 = [int(v) for v in box]
            label = f"{model.names[cls_id]} {score:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        output_image_path = os.path.join(output_dir, f"annotated_{image_path.name}")
        cv2.imwrite(output_image_path, image)
        
        # Store results
        results_data.append({
            "image": image_path.name,
            "detections": len(filtered_boxes),
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "classes": [model.names[c] for c in filtered_classes]
        })
    
    # Save results JSON
    results_json = os.path.join(output_dir, "results.json")
    with open(results_json, "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Save processing log
    log_json = os.path.join(output_dir, "processing_log.json")
    with open(log_json, "w") as f:
        json.dump(post_processor.processing_log, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print(f"Processed {len(results_data)} images successfully")


if __name__ == "__main__":
    # Configuration
    MODEL = "furniture_project/yolov11m_training/weights/best.pt"
    IMAGES = "test_images"
    OUTPUT = "inference_results"
    CONF_THRESHOLD = 0.35
    IOU_THRESHOLD = 0.45
    DEVICE = 0
    
    run_inference_with_postprocessing(
        model_path=MODEL,
        image_source=IMAGES,
        output_dir=OUTPUT,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        device=DEVICE
    )
