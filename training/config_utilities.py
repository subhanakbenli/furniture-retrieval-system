"""
Configuration and Utilities Module

Common configurations and utility functions for the training pipeline.

Contents:
- Default hyperparameters
- Path configurations
- Class definitions
- Logging utilities
- Data utilities
"""

import logging
from pathlib import Path
from datetime import datetime


# Logging Configuration
def setup_logging(log_dir="logs", log_name="training"):
    """
    Setup logging for pipeline
    
    Args:
        log_dir: Directory for log files
        log_name: Base name for log file
    
    Returns:
        Logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(log_name)


# Default Hyperparameters
DEFAULT_TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 16,
    "image_size": 640,
    "lr0": 0.01,                    # Initial learning rate
    "lrf": 0.01,                    # Final learning rate ratio
    "momentum": 0.937,              # Optimizer momentum
    "weight_decay": 0.0005,         # L2 regularization
    "patience": 10,                 # Early stopping patience
    "cache": True,                  # Cache images in memory
    "augment": True,                # Data augmentation
    "flipud": 0.5,                  # Vertical flip probability
    "fliplr": 0.5,                  # Horizontal flip probability
    "mosaic": 1.0,                  # Mosaic augmentation
}

INFERENCE_CONFIG = {
    "conf_threshold": 0.35,         # Confidence threshold
    "iou_threshold": 0.45,          # NMS IoU threshold
    "device": 0,                    # GPU device ID
}

POSTPROCESSING_CONFIG = {
    "iof_threshold": 0.85,          # IoF threshold for merging
    "conflict_iou_threshold": 0.97, # IoU for conflict detection
}


# Class Definitions
CLASS_NAMES = {
    0: "Furniture",
    1: "Textile",
    2: "Lighting",
    3: "Decoration",
    4: "Appliance"
}

CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}


# Path Configuration
class PathConfig:
    """Centralized path management"""
    
    def __init__(self, base_dir="."):
        self.base = Path(base_dir)
        
        # Datasets
        self.dataset_phase2 = self.base / "dataset_phase2_yolo_world"
        self.dataset_phase3 = self.base / "dataset_phase3_yolo_world"
        self.dataset_hard = self.base / "dataset_stage1_hard"
        
        # Models
        self.models_dir = self.base / "furniture_project"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Results
        self.results_dir = self.base / "results"
        self.inference_results = self.results_dir / "inference"
        self.test_results = self.results_dir / "test"
        self.logs_dir = self.results_dir / "logs"
        
        # Create directories
        for directory in [self.inference_results, self.test_results, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_data_yaml(self, dataset):
        """Get data.yaml path for dataset"""
        return getattr(self, f"dataset_{dataset}") / "data.yaml"


# Data Utilities
def validate_dataset_structure(dataset_path):
    """
    Validate dataset has required YOLO structure
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        True if valid, False otherwise
    """
    required_dirs = ["images", "labels"]
    
    dataset = Path(dataset_path)
    
    for required_dir in required_dirs:
        if not (dataset / required_dir).exists():
            return False
    
    # Check for train/val split
    for split in ["train", "val"]:
        if not (dataset / split / "images").exists():
            return False
        if not (dataset / split / "labels").exists():
            return False
    
    return True


def count_dataset_samples(dataset_path):
    """
    Count samples in YOLO dataset
    
    Args:
        dataset_path: Path to dataset
    
    Returns:
        Dictionary with sample counts
    """
    dataset = Path(dataset_path)
    
    counts = {
        "train": len(list((dataset / "train" / "images").glob("*"))),
        "val": len(list((dataset / "val" / "images").glob("*"))),
    }
    counts["total"] = counts["train"] + counts["val"]
    
    return counts


def get_gpu_info():
    """
    Get GPU information
    
    Returns:
        Dictionary with GPU details
    """
    try:
        import torch
        
        return {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        return {"cuda_available": False, "error": "PyTorch not installed"}


# Report Generation
def generate_training_report(config, results, output_path):
    """
    Generate training report
    
    Args:
        config: Training configuration
        results: Training results
        output_path: Output file path
    """
    
    report = f"""
Training Report
===============
Timestamp: {datetime.now().isoformat()}

Configuration:
- Epochs: {config.get('epochs', 'N/A')}
- Batch Size: {config.get('batch_size', 'N/A')}
- Image Size: {config.get('image_size', 'N/A')}
- Learning Rate: {config.get('lr0', 'N/A')}

Results:
- Total Duration: {results.get('duration', 'N/A')} hours
- Final Loss: {results.get('final_loss', 'N/A')}
- Best Accuracy: {results.get('best_accuracy', 'N/A')}
- Model Path: {results.get('model_path', 'N/A')}

Notes:
- Early stopping triggered if applicable
- Best model saved to weights/best.pt
"""
    
    with open(output_path, "w") as f:
        f.write(report)


if __name__ == "__main__":
    # Test utilities
    print("Configuration loaded successfully")
    print(f"Classes: {CLASS_NAMES}")
    print(f"GPU Info: {get_gpu_info()}")
