"""
Run the detector over every image in a folder through the running API.

Writes one JSON line per image to <out>/predictions.jsonl and, with --save-images,
the post-processed boxes drawn on each image to <out>/images/.

Usage:
    python scripts/predict_folder.py path/to/test_images --out results/ --save-images
    python scripts/predict_folder.py path/to/test_images --api http://localhost:8000 --conf 0.25 --iof 0.5

Only needs `requests` (and `pillow` for --save-images).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description="Batch detection through the /detect endpoint")
    parser.add_argument("images", type=Path, help="Folder with images (searched recursively)")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output folder")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iof", type=float, default=0.5, help="IoF threshold for same-class merging")
    parser.add_argument("--save-images", action="store_true", help="Also save images with boxes drawn")
    args = parser.parse_args()

    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        sys.exit(f"No images found in {args.images}")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (args.out / "images").mkdir(exist_ok=True)

    data = {"conf_threshold": args.conf, "iof_threshold": args.iof}
    n_boxes, failed, start = 0, 0, time.time()

    with open(args.out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, path in enumerate(paths, 1):
            rel = path.relative_to(args.images)
            try:
                with open(path, "rb") as img:
                    resp = requests.post(f"{args.api}/detect", files={"file": (path.name, img)}, data=data, timeout=120)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(paths)}] {rel}: FAILED ({e})")
                continue

            result["filename"] = str(rel)
            f.write(json.dumps(result) + "\n")
            n_boxes += len(result["detections"])
            print(f"[{i}/{len(paths)}] {rel}: {len(result['detections'])} objects")

            if args.save_images:
                save_annotated(path, result["detections"], args.out / "images" / rel)

    elapsed = time.time() - start
    print(f"\nDone: {len(paths) - failed} images, {n_boxes} objects, {failed} failed, {elapsed:.1f}s")
    print(f"Predictions: {args.out / 'predictions.jsonl'}")


def save_annotated(path, detections, out_path):
    from PIL import Image, ImageDraw

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for det in detections:
        box = det["box"]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 3), f"{det['label']} {det['confidence']:.2f}", fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path.with_suffix(".jpg"), quality=90)


if __name__ == "__main__":
    main()
