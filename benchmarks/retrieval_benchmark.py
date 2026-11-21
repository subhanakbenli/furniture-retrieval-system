"""
Room-level retrieval benchmark on synthetic rooms.

Dataset layout (not distributed with this repo):
    <data>/dataset.json          catalog: [{object_id, category, embedding (512-d CLIP), ...}]
    <data>/room_final/rooms.json rooms:   [{room_id, object_ids, image_path, ...}]
    <data>/room_final/*.png      room images (each contains the listed catalog products)

Protocol (per room):
    1. detect objects (Golden.pt + post-processing, or a YOLO-World baseline)
    2. crop each detection and embed it with CLIP ViT-B/32 (same model as the catalog)
    3. search the catalog products of the detected category (cosine, top-K, score >= threshold)
    4. room recall    = |retrieved ∩ ground truth| / |ground truth|
       room precision = |retrieved ∩ ground truth| / |retrieved|
Metrics are averaged over rooms. Recall is also reported at K = 1, 3, 5, 10.

Because every crop retrieves K products from its category, recall grows with the
number of crops. As a control, `random_recall@K` is the expected recall if each crop
returned K random products of its detected category instead of the CLIP top-K.

Usage:
    python benchmarks/retrieval_benchmark.py --data path/to/dataset --detector golden --out results/golden
    python benchmarks/retrieval_benchmark.py --data path/to/dataset --detector yoloworld --out results/yoloworld
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "service"))
from src.postprocess import PostProcessor  # noqa: E402

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CATEGORY_ALIASES = {"Lightning": "Lighting"}  # the catalog file spells the class "Lightning"
K_VALUES = (1, 3, 5, 10)

# Open-vocabulary prompts used for pseudo-labelling, grouped into the 5 superclasses
YOLO_WORLD_HIERARCHY = {
    "Furniture": ["sofa", "couch", "armchair", "loveseat", "chair", "dining chair", "office chair", "stool", "bench", "ottoman", "pouf", "dining table", "coffee table", "side table", "nightstand", "bedside table", "desk", "computer desk", "console table", "wardrobe", "closet", "cabinet", "cupboard", "chest of drawers", "dresser", "bookshelf", "bookcase", "TV stand", "media unit", "bed frame", "headboard", "coat rack", "shoe rack"],
    "Textile": ["rug", "carpet", "floor mat", "runner", "doormat", "curtain", "drape", "window blind", "pillow", "cushion", "throw pillow", "blanket", "throw blanket", "duvet", "quilt", "comforter", "bed spread", "messy bed sheets", "bedding", "crumpled blanket", "towel", "bath mat"],
    "Lighting": ["floor lamp", "standing lamp", "reading lamp", "table lamp", "desk lamp", "bedside lamp", "ceiling light", "pendant light", "chandelier", "wall sconce", "wall lamp", "lampshade"],
    "Decoration": ["vase", "flower pot", "planter", "potted plant", "indoor plant", "mirror", "wall mirror", "standing mirror", "picture frame", "photo frame", "painting", "wall art", "poster", "sculpture", "statue", "figurine", "bust", "candle", "candlestick", "candle holder", "decorative bowl", "tray", "basket", "storage box", "wall clock", "table clock"],
    "Appliance": ["television", "tv screen", "monitor", "display", "refrigerator", "fridge", "washing machine", "dryer", "dishwasher", "oven", "stove", "microwave", "toaster", "kettle", "coffee maker", "speaker", "soundbar", "air conditioner", "fan", "heater", "radiator"],
}


class Detector:
    def __init__(self, kind, weights, conf, iou, iof):
        from ultralytics import YOLO, YOLOWorld

        self.kind, self.conf, self.iou = kind, conf, iou
        if kind == "golden":
            self.model = YOLO(weights)
            self.label_of = lambda cls: self.model.names[cls]
            self.post = PostProcessor(iof_threshold=iof)
        else:
            self.model = YOLOWorld(weights)
            prompts, supers = [], []
            for sup, items in YOLO_WORLD_HIERARCHY.items():
                prompts += items
                supers += [sup] * len(items)
            self.model.set_classes(prompts)
            self.label_of = lambda cls: supers[cls]
            self.post = None

    def __call__(self, image):
        res = self.model(image, conf=self.conf, iou=self.iou, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes else []
        scores = res.boxes.conf.cpu().numpy().tolist() if res.boxes else []
        classes = res.boxes.cls.cpu().numpy().astype(int).tolist() if res.boxes else []
        labels = [self.label_of(c) for c in classes]
        if self.post is not None:
            boxes, scores, classes = self.post.apply_rules(boxes, scores, classes)
            labels = [self.label_of(c) for c in classes]
        return list(zip(boxes, scores, labels)), res.speed["inference"]


def load_catalog(path):
    items = json.load(open(path))
    by_cat = defaultdict(lambda: ([], []))
    for it in items:
        cat = CATEGORY_ALIASES.get(it["category"], it["category"])
        emb = np.asarray(it["embedding"], dtype=np.float32)
        by_cat[cat][0].append(emb / np.linalg.norm(emb))
        by_cat[cat][1].append(it["object_id"])
    catalog = {c: (np.vstack(e), np.asarray(ids)) for c, (e, ids) in by_cat.items()}
    category_of = {it["object_id"]: CATEGORY_ALIASES.get(it["category"], it["category"]) for it in items}
    return catalog, category_of


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True, help="Folder with dataset.json and room_final/")
    p.add_argument("--detector", choices=["golden", "yoloworld"], default="golden")
    p.add_argument("--weights", default=None, help="Detector weights (default: service/weights/Golden.pt or yolov8l-worldv2.pt)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    p.add_argument("--iof", type=float, default=0.5, help="Same-class merge threshold (golden only)")
    p.add_argument("--min-area", type=int, default=800, help="Drop crops smaller than this many pixels")
    p.add_argument("--k", type=int, default=5, help="Top-K per crop for the headline metrics")
    p.add_argument("--threshold", type=float, default=0.2, help="Minimum cosine similarity")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--limit", type=int, default=None, help="Only the first N rooms (smoke test)")
    args = p.parse_args()

    weights = args.weights or (str(REPO / "service/weights/Golden.pt") if args.detector == "golden" else "yolov8l-worldv2.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    catalog, category_of = load_catalog(args.data / "dataset.json")
    rooms = json.load(open(args.data / "room_final" / "rooms.json"))[: args.limit]
    detector = Detector(args.detector, weights, args.conf, args.iou, args.iof)
    clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    k_max = max(K_VALUES + (args.k,))
    per_room, latencies = [], []
    gt_hits = defaultdict(lambda: [0, 0])  # category -> [found at K, total]

    for i, room in enumerate(rooms, 1):
        image_path = args.data / "room_final" / Path(room["image_path"]).name
        image = Image.open(image_path).convert("RGB")
        gt = set(room["object_ids"])

        detections, ms = detector(image)
        latencies.append(ms)
        crops, cats = [], []
        for box, _, label in detections:
            x1, y1, x2, y2 = [int(v) for v in box]
            if (x2 - x1) * (y2 - y1) < args.min_area or label not in catalog:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            cats.append(label)

        # retrieved ids per K over all crops of the room
        retrieved = {k: set() for k in set(K_VALUES + (args.k,))}
        if crops:
            with torch.no_grad():
                inputs = processor(images=crops, return_tensors="pt").to(device)
                # vision tower + projection (= get_image_features, stable across transformers versions)
                pooled = clip.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
                feats = clip.visual_projection(pooled)
                feats = (feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy()
            for feat, cat in zip(feats, cats):
                embs, ids = catalog[cat]
                sims = embs @ feat
                order = np.argsort(-sims)[:k_max]
                for k in retrieved:
                    retrieved[k].update(int(ids[j]) for j in order[:k] if sims[j] >= args.threshold)

        # expected recall if each crop picked K random products of its category
        crops_per_cat = defaultdict(int)
        for c in cats:
            crops_per_cat[c] += 1
        rand = {}
        for k in K_VALUES:
            probs = []
            for oid in gt:
                c = category_of[oid]
                n = len(catalog[c][1]) if c in catalog else 1
                probs.append(1 - (1 - min(k, n) / n) ** crops_per_cat.get(c, 0))
            rand[k] = float(np.mean(probs))

        row = {"room_id": room["room_id"], "room_type": room.get("room_type"), "n_gt": len(gt),
               "n_detections": len(detections), "n_crops": len(crops), "latency_ms": round(ms, 1)}
        for k, found in retrieved.items():
            hit = len(found & gt)
            row[f"recall@{k}"] = hit / len(gt)
            row[f"precision@{k}"] = hit / len(found) if found else 0.0
            row[f"retrieved@{k}"] = len(found)
        for k in K_VALUES:
            row[f"random_recall@{k}"] = rand[k]
        for oid in gt:
            gt_hits[category_of[oid]][1] += 1
            gt_hits[category_of[oid]][0] += oid in retrieved[args.k]
        per_room.append(row)
        if i % 25 == 0 or i == len(rooms):
            print(f"[{i}/{len(rooms)}] mean recall@{args.k} = {np.mean([r[f'recall@{args.k}'] for r in per_room]):.3f}")

    mean = lambda key: float(np.mean([r[key] for r in per_room]))
    summary = {
        "detector": args.detector,
        "weights": Path(weights).name,
        "settings": {"conf": args.conf, "nms_iou": args.iou, "iof": args.iof, "min_area": args.min_area,
                     "k": args.k, "similarity_threshold": args.threshold, "clip": CLIP_MODEL_ID},
        "rooms": len(per_room),
        f"recall@{args.k}": mean(f"recall@{args.k}"),
        f"precision@{args.k}": mean(f"precision@{args.k}"),
        "recall_at": {k: mean(f"recall@{k}") for k in K_VALUES},
        "precision_at": {k: mean(f"precision@{k}") for k in K_VALUES},
        "random_recall_at": {k: mean(f"random_recall@{k}") for k in K_VALUES},
        "mean_detections_per_room": mean("n_detections"),
        "mean_crops_per_room": mean("n_crops"),
        f"mean_retrieved_per_room@{args.k}": mean(f"retrieved@{args.k}"),
        f"recall@{args.k}_by_category": {c: round(h / t, 3) for c, (h, t) in sorted(gt_hits.items())},
        "detector_latency_ms": {"mean": float(np.mean(latencies)), "median": float(np.median(latencies)),
                                "device": device},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out / "summary.json", "w"), indent=2)
    with open(args.out / "per_room.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_room[0]))
        w.writeheader()
        w.writerows(per_room)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
