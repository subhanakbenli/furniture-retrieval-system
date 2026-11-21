"""
Data Mining and Hard Example Selection

This module implements intelligent sampling strategies to select training data.
It identifies images with specific characteristics useful for improving model performance.

Sampling Categories:
1. Weak Detections: Objects detected with low-medium confidence (15-55%)
2. Context Confusion: Images with both target and confusing objects
3. Negative Samples: Images with no or minimal target objects

Strategy:
- Stochastic sampling from image pool
- Per-category quotas to maintain balanced dataset
- Confidence-based filtering
- Multi-object relationship analysis
"""

import csv
import random
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json


def mine_hard_examples_stochastic(
    model_path,
    image_pool_dir,
    output_csv,
    target_count=6000,
    quota_weak=0.60,
    quota_context=0.20,
    quota_negative=0.20,
    furniture_class_id=0,
    lighting_class_id=2,
    device=0
):
    """
    Stochastically mine hard examples from image pool
    
    Args:
        model_path: Path to trained model
        image_pool_dir: Directory containing candidate images
        output_csv: Output CSV file path
        target_count: Total target number of examples
        quota_weak: Fraction for weak detections
        quota_context: Fraction for context confusion
        quota_negative: Fraction for negatives
        furniture_class_id: Class ID for furniture
        lighting_class_id: Class ID for lighting
        device: GPU device ID or 'cpu'
    """
    
    # Quotas
    quotas = {
        "weak": int(target_count * quota_weak),
        "context": int(target_count * quota_context),
        "negative": int(target_count * quota_negative),
    }
    
    counts = {"weak": 0, "context": 0, "negative": 0}
    selected_images = set()
    output_rows = []
    
    print(f"Mining hard examples from {image_pool_dir}")
    print(f"Target: {target_count}")
    print(f"Quotas: {quotas}")
    
    # Load model
    model = YOLO(model_path)
    
    # Collect image pool
    image_pool = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_pool.extend(Path(image_pool_dir).rglob(ext))
    
    print(f"Image pool size: {len(image_pool)}")
    
    # Stochastic sampling loop
    attempts = 0
    max_attempts = target_count * 20
    
    pbar = tqdm(total=target_count, desc="Collected")
    
    while sum(counts.values()) < target_count and attempts < max_attempts:
        attempts += 1
        
        # Random image selection
        image_path = random.choice(image_pool)
        image_str = str(image_path)
        
        if image_str in selected_images:
            continue
        
        try:
            # Run inference
            results = model.predict(
                image_str,
                conf=0.15,
                iou=0.5,
                device=device,
                verbose=False
            )
            
            if not results or len(results) == 0:
                continue
            
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            # Analyze detections
            has_lighting = False
            has_furniture = False
            max_lighting_conf = 0.0
            
            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                
                if cls == lighting_class_id:
                    has_lighting = True
                    max_lighting_conf = max(max_lighting_conf, conf)
                elif cls == furniture_class_id:
                    has_furniture = True
            
            category = None
            
            # Category 1: Weak lighting detections
            if (has_lighting and 0.15 <= max_lighting_conf <= 0.55 and
                counts["weak"] < quotas["weak"]):
                category = "weak"
            
            # Category 2: Context confusion (both lighting and furniture)
            elif (has_lighting and has_furniture and
                  counts["context"] < quotas["context"]):
                category = "context"
            
            # Category 3: Negative samples (no lighting)
            elif (not has_lighting and has_furniture and
                  counts["negative"] < quotas["negative"]):
                category = "negative"
            
            # Add to results if category matched
            if category:
                selected_images.add(image_str)
                counts[category] += 1
                pbar.update(1)
                
                output_rows.append({
                    "path": image_str,
                    "category": category,
                    "detections": len(boxes),
                    "max_confidence": max_lighting_conf
                })
        
        except Exception as e:
            continue
    
    pbar.close()
    
    # Write CSV
    if output_rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["path", "category", "detections", "max_confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
    
    # Print summary
    print(f"Mining complete!")
    print(f"Total selected: {sum(counts.values())}")
    print(f"Weak: {counts['weak']}")
    print(f"Context: {counts['context']}")
    print(f"Negative: {counts['negative']}")
    print(f"Results saved to {output_csv}")


def analyze_detection_patterns(model_path, image_dir, output_json, device=0):
    """
    Analyze detection patterns across images to understand model behavior
    
    Args:
        model_path: Path to model
        image_dir: Directory of images to analyze
        output_json: Output JSON file
        device: GPU device ID or 'cpu'
    """
    
    model = YOLO(model_path)
    image_list = list(Path(image_dir).rglob("*.jpg")) + \
                 list(Path(image_dir).rglob("*.png"))
    
    analysis = {
        "total_images": len(image_list),
        "class_distribution": {},
        "confidence_stats": {},
        "detection_rate": 0.0,
        "avg_detections_per_image": 0.0
    }
    
    total_detections = 0
    images_with_detections = 0
    
    for image_path in tqdm(image_list, desc="Analyzing"):
        results = model.predict(str(image_path), device=device, verbose=False)
        
        if results and results[0].boxes:
            images_with_detections += 1
            boxes = results[0].boxes
            
            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                class_name = model.names[cls]
                
                if class_name not in analysis["class_distribution"]:
                    analysis["class_distribution"][class_name] = 0
                analysis["class_distribution"][class_name] += 1
                
                if class_name not in analysis["confidence_stats"]:
                    analysis["confidence_stats"][class_name] = []
                analysis["confidence_stats"][class_name].append(conf)
                
                total_detections += 1
    
    # Compute statistics
    analysis["detection_rate"] = images_with_detections / len(image_list)
    analysis["avg_detections_per_image"] = (
        total_detections / images_with_detections if images_with_detections > 0 else 0
    )
    
    # Save analysis
    with open(output_json, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis saved to {output_json}")


if __name__ == "__main__":
    # Configuration
    MODEL = "furniture_project/stage2/weights/best.pt"
    IMAGE_POOL = "images"
    OUTPUT_CSV = "hard_examples_mined.csv"
    
    mine_hard_examples_stochastic(
        model_path=MODEL,
        image_pool_dir=IMAGE_POOL,
        output_csv=OUTPUT_CSV,
        target_count=6000,
        quota_weak=0.60,
        quota_context=0.20,
        quota_negative=0.20
    )
