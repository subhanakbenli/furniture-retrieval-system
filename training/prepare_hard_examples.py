"""
Hard Example Finder - Identifies challenging samples for model training

This script extracts hard examples from a CSV file and builds a dedicated dataset.
Hard examples are images where the model might struggle or perform poorly.

Process:
1. Reads CSV with hard example paths
2. Locates corresponding images and labels
3. Copies them to a new dataset directory
4. Generates data.yaml for training

Configuration:
- CSV_PATH: CSV file containing hard example paths
- BASE_DATASET: Source dataset directory
- STAGE1_DATASET: Output directory for hard examples

"""

import csv
import shutil
from pathlib import Path
import yaml


def find_image_and_label(rel_path, train_img, train_lbl, val_img, val_lbl):
    """
    Locate image and label files from relative path
    
    Args:
        rel_path: Relative path from CSV
        train_img, train_lbl: Training set paths
        val_img, val_lbl: Validation set paths
    
    Returns:
        Tuple of (image_path, label_path) or (None, None) if not found
    """
    img_name = Path(rel_path).name
    lbl_name = img_name.replace(".jpg", ".txt")

    if (train_img / img_name).exists():
        return train_img / img_name, train_lbl / lbl_name
    if (val_img / img_name).exists():
        return val_img / img_name, val_lbl / lbl_name

    return None, None


def build_hard_example_dataset(csv_path, base_dataset_path, output_dataset_path):
    """
    Build a dataset containing only hard examples
    
    Args:
        csv_path: Path to CSV file with hard example entries
        base_dataset_path: Source dataset directory
        output_dataset_path: Target directory for hard examples
    """
    
    base_dataset = Path(base_dataset_path)
    stage1_dataset = Path(output_dataset_path)
    
    train_img = base_dataset / "train" / "images"
    train_lbl = base_dataset / "train" / "labels"
    val_img = base_dataset / "val" / "images"
    val_lbl = base_dataset / "val" / "labels"
    
    (stage1_dataset / "images").mkdir(parents=True, exist_ok=True)
    (stage1_dataset / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row["path"]

            img_src, lbl_src = find_image_and_label(
                rel_path, train_img, train_lbl, val_img, val_lbl
            )

            if img_src is None or lbl_src is None or not lbl_src.exists():
                missing += 1
                continue

            shutil.copy(img_src, stage1_dataset / "images" / img_src.name)
            shutil.copy(lbl_src, stage1_dataset / "labels" / lbl_src.name)
            copied += 1

    print(f"Copied: {copied} files")
    print(f"Missing: {missing} files")


def create_data_yaml(dataset_path, output_yaml_path, classes):
    """
    Generate data.yaml for YOLO training
    
    Args:
        dataset_path: Root dataset directory
        output_yaml_path: Output YAML file path
        classes: Dictionary of class_id: class_name
    """
    
    data_dict = {
        "path": str(Path(dataset_path).absolute()),
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": len(classes),
        "names": classes
    }
    
    with open(output_yaml_path, "w") as f:
        yaml.dump(data_dict, f, default_flow_style=False)
    
    print(f"Created data.yaml: {output_yaml_path}")


if __name__ == "__main__":
    # Configuration
    CSV_PATH = "hard_examples_smart.csv"
    BASE_DATASET = "dataset_phase2_yolo_world"
    OUTPUT_DATASET = "dataset_stage1_hard"
    
    # Classes matching your furniture detection model
    CLASSES = {
        0: "Furniture",
        1: "Textile",
        2: "Lighting",
        3: "Decoration",
        4: "Appliance"
    }
    
    print("Starting hard example dataset extraction...")
    build_hard_example_dataset(CSV_PATH, BASE_DATASET, OUTPUT_DATASET)
    create_data_yaml(OUTPUT_DATASET, f"{OUTPUT_DATASET}/data.yaml", CLASSES)
    print("Done!")
