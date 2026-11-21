"""
Model Training Script for Furniture Detection

This script handles the complete training pipeline for YOLO-based furniture detection.

Features:
- Multiple model size options (Nano, Small, Medium, Large)
- Configurable training parameters
- Automatic model checkpointing
- CUDA/CPU device selection

Model Size Guide:
- Nano (yolov8n.pt):   Fastest, lowest accuracy - suitable for mobile/edge
- Small (yolov8s.pt):  Fast, moderate accuracy - good balance
- Medium (yolov8m.pt): Standard, good accuracy - recommended
- Large (yolov8l.pt):  Slow, highest accuracy - for powerful hardware

Training Parameters:
- EPOCHS: Number of complete passes through dataset (50-100 recommended)
- BATCH_SIZE: Images per iteration (reduce if out of memory)
- IMAGE_SIZE: Input resolution (640 is standard for YOLO)
- PATIENCE: Early stopping after N epochs without improvement
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path


def train_furniture_detection_model(
    data_yaml_path,
    model_path,
    epochs=50,
    batch_size=16,
    image_size=640,
    device=0,
    project_name="furniture_project",
    experiment_name="yolov11m_training"
):
    """
    Train a YOLO furniture detection model
    
    Args:
        data_yaml_path: Path to data.yaml configuration file
        model_path: Path to model weights (pre-trained or new)
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image resolution
        device: GPU device ID (0) or CPU ('cpu')
        project_name: Project directory for results
        experiment_name: Name for this training run
    
    Returns:
        Training results object
    """
    
    # Validate configuration
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    print(f"Starting model training...")
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml_path}")
    print(f"Config: epochs={epochs}, batch={batch_size}, size={image_size}")
    
    # Load model
    model = YOLO(model_path)
    
    # Training configuration
    training_config = {
        "data": data_yaml_path,
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "patience": 10,              # Early stopping patience
        "save": True,                # Save best model
        "cache": True,               # Cache images in RAM for speed
        "project": project_name,
        "name": experiment_name,
        "verbose": True,
        "lr0": 0.01,                 # Initial learning rate
        "lrf": 0.01,                 # Final learning rate ratio
        "momentum": 0.937,           # Optimizer momentum
        "weight_decay": 0.0005,      # L2 regularization
        "augment": True,             # Use data augmentation
        "flipud": 0.5,               # Vertical flip probability
        "fliplr": 0.5,               # Horizontal flip probability
        "mosaic": 1.0,               # Mosaic augmentation
    }
    
    # Train model
    results = model.train(**training_config)
    
    # Print summary
    print("Training completed!")
    print(f"Results directory: {project_name}/{experiment_name}")
    print(f"Best model: {project_name}/{experiment_name}/weights/best.pt")
    
    return results


def validate_model(model_path, data_yaml_path, device=0):
    """
    Validate trained model on validation set
    
    Args:
        model_path: Path to trained model weights
        data_yaml_path: Path to data.yaml
        device: GPU device ID or 'cpu'
    
    Returns:
        Validation metrics
    """
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path, device=device)
    return metrics


def export_model(model_path, export_format="onnx"):
    """
    Export trained model to different formats
    
    Args:
        model_path: Path to trained model
        export_format: Export format (onnx, torchscript, tflite, etc.)
    
    Returns:
        Path to exported model
    """
    model = YOLO(model_path)
    export_path = model.export(format=export_format)
    return export_path


if __name__ == "__main__":
    # Configuration
    DATA_YAML = "dataset_phase2_yolo_world/data.yaml"
    MODEL_WEIGHTS = "yolov11m.pt"  # Pre-trained YOLOv11 Medium
    
    EPOCHS = 50
    BATCH_SIZE = 16
    IMAGE_SIZE = 640
    DEVICE = 0  # GPU 0, or 'cpu'
    PROJECT = "furniture_project"
    EXPERIMENT = "yolov11m_detection_v1"
    
    # Train model
    results = train_furniture_detection_model(
        data_yaml_path=DATA_YAML,
        model_path=MODEL_WEIGHTS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        device=DEVICE,
        project_name=PROJECT,
        experiment_name=EXPERIMENT
    )
    
    # Validate
    model_path = f"{PROJECT}/{EXPERIMENT}/weights/best.pt"
    validate_model(model_path, DATA_YAML, DEVICE)
    
    # Optional: Export to ONNX format
    # export_model(model_path, "onnx")
