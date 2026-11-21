"""
Testing and Evaluation Module

Comprehensive testing framework for trained models including:
- Batch inference on test sets
- Metric computation
- Result visualization
- Performance analysis

Test Configuration:
- Model selection
- Input images directory
- Output results directory
- Confidence and IoU thresholds
- Visualization options
"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json
from datetime import datetime


def run_comprehensive_test(
    model_path,
    test_images_dir,
    output_dir,
    conf_threshold=0.35,
    iou_threshold=0.45,
    visualize=True,
    device=0
):
    """
    Run comprehensive testing on trained model
    
    Args:
        model_path: Path to trained model weights
        test_images_dir: Directory containing test images
        output_dir: Directory for results
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        visualize: Whether to save annotated images
        device: GPU device ID or 'cpu'
    """
    
    # Validation
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Test directory not found: {test_images_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting comprehensive model testing...")
    print(f"Model: {model_path}")
    print(f"Test set: {test_images_dir}")
    
    # Load model
    model = YOLO(model_path)
    
    # Collect test images
    image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    test_images = []
    for ext in image_extensions:
        test_images.extend(Path(test_images_dir).rglob(f"*{ext}"))
    
    print(f"Found {len(test_images)} test images")
    
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "config": {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold
        },
        "total_images": len(test_images),
        "total_detections": 0,
        "class_distribution": {},
        "images_with_detections": 0,
        "detailed_results": []
    }
    
    # Process images
    for image_path in tqdm(test_images, desc="Testing"):
        try:
            # Inference
            results = model.predict(
                str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False
            )
            
            if not results:
                continue
            
            result = results[0]
            
            # Extract detections
            image_result = {
                "image": image_path.name,
                "detections": [],
                "count": 0
            }
            
            if result.boxes is not None:
                image_result["count"] = len(result.boxes)
                results_summary["images_with_detections"] += 1
                results_summary["total_detections"] += len(result.boxes)
                
                # Visualize if requested
                if visualize:
                    image = cv2.imread(str(image_path))
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        cls_id = int(box.cls)
                        class_name = model.names[cls_id]
                        
                        # Update distribution
                        if class_name not in results_summary["class_distribution"]:
                            results_summary["class_distribution"][class_name] = 0
                        results_summary["class_distribution"][class_name] += 1
                        
                        # Record detection
                        image_result["detections"].append({
                            "class": class_name,
                            "confidence": conf,
                            "box": [float(x1), float(y1), float(x2), float(y2)]
                        })
                        
                        # Draw on image
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(
                            image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )
                    
                    # Save annotated image
                    output_path = os.path.join(output_dir, f"test_{image_path.name}")
                    cv2.imwrite(output_path, image)
            
            results_summary["detailed_results"].append(image_result)
        
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    # Compute statistics
    if results_summary["images_with_detections"] > 0:
        results_summary["avg_detections"] = (
            results_summary["total_detections"] /
            results_summary["images_with_detections"]
        )
    else:
        results_summary["avg_detections"] = 0
    
    results_summary["detection_rate"] = (
        results_summary["images_with_detections"] /
        results_summary["total_images"]
    )
    
    # Save summary
    summary_path = os.path.join(output_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    print(f"Total images: {results_summary['total_images']}")
    print(f"Images with detections: {results_summary['images_with_detections']}")
    print(f"Detection rate: {results_summary['detection_rate']:.2%}")
    print(f"Total detections: {results_summary['total_detections']}")
    print(f"Avg detections per positive: {results_summary['avg_detections']:.2f}")
    print(f"Class distribution: {results_summary['class_distribution']}")
    print(f"Results saved to: {output_dir}")
    
    return results_summary


if __name__ == "__main__":
    # Configuration
    MODEL = "furniture_project/yolov11m_training/weights/best.pt"
    TEST_DIR = "test_images"
    OUTPUT = "test_results"
    
    run_comprehensive_test(
        model_path=MODEL,
        test_images_dir=TEST_DIR,
        output_dir=OUTPUT,
        conf_threshold=0.35,
        iou_threshold=0.45,
        visualize=True,
        device=0
    )
