"""
Controlled comparison of Golden.pt and a YOLO-World baseline as the detector of the
retrieval pipeline, on the synthetic-room set used by retrieval_benchmark.py.

Room-level recall rewards detectors that output more boxes (every crop retrieves K
products). This script removes that advantage:

  * budget:   each detector keeps only its N most confident crops per room
  * sweep:    confidence thresholds 0.05 .. 0.5 -> recall vs. crops-per-room curves
  * hit rate: fraction of crops whose top-1 product is in the room (independent of count)
  * paired bootstrap over rooms for the recall difference at each budget (95% CI)

YOLO-World is also evaluated with class-agnostic NMS, which removes duplicate boxes
from overlapping prompts (e.g. "sofa" and "couch") and is its strongest variant here.

Detections are computed once at conf 0.05 and cached; all settings are evaluated
from the cache.

Usage:
    python benchmarks/detector_comparison.py --data path/to/dataset --out results/comparison
"""
import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_benchmark import CLIP_MODEL_ID, REPO, YOLO_WORLD_HIERARCHY, load_catalog  # noqa: E402
from src.postprocess import PostProcessor  # noqa: E402

BASE_CONF, NMS_IOU, MIN_AREA, K, SIM_THRESHOLD = 0.05, 0.6, 800, 5, 0.2
BUDGETS = (3, 5, 8, 10, 15, 20)
CONFS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def run_detectors(rooms, data_dir, cache_path):
    if cache_path.exists():
        return pickle.load(open(cache_path, "rb"))
    from ultralytics import YOLO, YOLOWorld

    golden = YOLO(str(REPO / "service/weights/Golden.pt"))
    world = YOLOWorld("yolov8l-worldv2.pt")
    prompts, supers = [], []
    for sup, items in YOLO_WORLD_HIERARCHY.items():
        prompts += items
        supers += [sup] * len(items)
    world.set_classes(prompts)

    cache = {"golden": {}, "yoloworld": {}, "latency": defaultdict(list)}
    for i, room in enumerate(rooms, 1):
        image = Image.open(data_dir / "room_final" / Path(room["image_path"]).name).convert("RGB")
        for name, model, names in (("golden", golden, None), ("yoloworld", world, supers)):
            res = model(image, conf=BASE_CONF, iou=NMS_IOU, verbose=False)[0]
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cache[name][room["room_id"]] = {
                "boxes": res.boxes.xyxy.cpu().numpy(),
                "scores": res.boxes.conf.cpu().numpy(),
                "cls": cls,
                "labels": [model.names[c] if names is None else names[c] for c in cls],
            }
            cache["latency"][name].append(res.speed["inference"])
        if i % 50 == 0:
            print(f"detected {i}/{len(rooms)} rooms")
    cache["latency"] = dict(cache["latency"])
    pickle.dump(cache, open(cache_path, "wb"))
    return cache


def select_crops(det, variant, conf, budget):
    """Boxes/labels for one room under a given setting, most confident first."""
    keep = det["scores"] >= conf
    boxes, scores, cls = det["boxes"][keep], det["scores"][keep], det["cls"][keep]
    labels = [l for l, k in zip(det["labels"], keep) if k]

    if variant == "golden" and len(boxes):
        b, s, c = PostProcessor().apply_rules(boxes.tolist(), scores.tolist(), cls.tolist())
        lab_of = dict(zip(cls.tolist(), labels))
        boxes, scores, labels = np.asarray(b), np.asarray(s), [lab_of[x] for x in c]
    elif variant == "yoloworld_agnostic" and len(boxes):
        idx = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), NMS_IOU).numpy()
        boxes, scores, labels = boxes[idx], scores[idx], [labels[j] for j in idx]

    out = []
    for j in np.argsort(-np.asarray(scores)) if len(scores) else []:
        x1, y1, x2, y2 = (int(v) for v in boxes[j])
        if (x2 - x1) * (y2 - y1) >= MIN_AREA:
            out.append(((x1, y1, x2, y2), labels[j]))
    return out[:budget] if budget else out


class Embedder:
    def __init__(self, device):
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.cache = {}

    def __call__(self, room_id, image_loader, boxes):
        todo = [b for b in boxes if (room_id, b) not in self.cache]
        if todo:
            image = image_loader()
            with torch.no_grad():
                px = self.proc(images=[image.crop(b) for b in todo], return_tensors="pt")["pixel_values"].to(self.device)
                f = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
                f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
            for b, v in zip(todo, f):
                self.cache[(room_id, b)] = v
        return [self.cache[(room_id, b)] for b in boxes]


def evaluate(rooms, crops_by_room, embed, catalog, category_of, data_dir):
    """Per-room recall@1, recall@5, crop hit rate, random-control recall@5."""
    rows = []
    for room in rooms:
        rid, gt = room["room_id"], set(room["object_ids"])
        crops = [(b, l) for b, l in crops_by_room[rid] if l in catalog]
        found1, found5, hits, per_cat = set(), set(), 0, defaultdict(int)
        if crops:
            loader = lambda: Image.open(data_dir / "room_final" / Path(room["image_path"]).name).convert("RGB")
            feats = embed(rid, loader, [b for b, _ in crops])
            for (b, cat), f in zip(crops, feats):
                embs, ids = catalog[cat]
                sims = embs @ f
                order = np.argsort(-sims)[:K]
                order = [j for j in order if sims[j] >= SIM_THRESHOLD]
                if order:
                    found1.add(int(ids[order[0]]))
                    hits += int(ids[order[0]]) in gt
                found5.update(int(ids[j]) for j in order)
                per_cat[cat] += 1
        rand = np.mean([1 - (1 - min(K, len(catalog[category_of[o]][1])) / len(catalog[category_of[o]][1])) ** per_cat.get(category_of[o], 0) for o in gt])
        rows.append({"r1": len(found1 & gt) / len(gt), "r5": len(found5 & gt) / len(gt),
                     "crops": len(crops), "hits": hits, "rand5": rand})
    n_crops = sum(r["crops"] for r in rows)
    return {
        "recall@1": float(np.mean([r["r1"] for r in rows])),
        "recall@5": float(np.mean([r["r5"] for r in rows])),
        "random_recall@5": float(np.mean([r["rand5"] for r in rows])),
        "crops_per_room": n_crops / len(rows),
        "crop_hit_rate": sum(r["hits"] for r in rows) / n_crops if n_crops else 0.0,
        "_r5": [r["r5"] for r in rows],
        "_r1": [r["r1"] for r in rows],
    }


def paired_bootstrap(a, b, n=10000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/comparison"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog, category_of = load_catalog(args.data / "dataset.json")
    rooms = json.load(open(args.data / "room_final" / "rooms.json"))
    cache = run_detectors(rooms, args.data, args.out / "detections.pkl")
    embed = Embedder(device)

    variants = {"golden": "golden", "yoloworld": "yoloworld", "yoloworld_agnostic": "yoloworld"}
    results = {"budget": {}, "sweep": {}}

    t = time.time()
    for v, src in variants.items():
        results["budget"][v] = {}
        for n in BUDGETS:
            crops = {rid: select_crops(d, v, BASE_CONF, n) for rid, d in cache[src].items()}
            results["budget"][v][n] = evaluate(rooms, crops, embed, catalog, category_of, args.data)
        results["sweep"][v] = {}
        for c in CONFS:
            crops = {rid: select_crops(d, v, c, None) for rid, d in cache[src].items()}
            results["sweep"][v][c] = evaluate(rooms, crops, embed, catalog, category_of, args.data)
        print(f"evaluated {v} ({time.time() - t:.0f}s)")

    # paired differences Golden - YOLO-World (both variants) at each budget
    results["paired_diff_recall@5"] = {
        base: {n: paired_bootstrap(results["budget"]["golden"][n]["_r5"], results["budget"][base][n]["_r5"]) for n in BUDGETS}
        for base in ("yoloworld", "yoloworld_agnostic")
    }
    results["paired_diff_recall@1"] = {
        base: {n: paired_bootstrap(results["budget"]["golden"][n]["_r1"], results["budget"][base][n]["_r1"]) for n in BUDGETS}
        for base in ("yoloworld", "yoloworld_agnostic")
    }
    results["latency_ms_median"] = {k: float(np.median(v)) for k, v in cache["latency"].items()}
    results["device"] = device

    strip = lambda d: {k: strip(v) for k, v in d.items() if not str(k).startswith("_")} if isinstance(d, dict) else d
    json.dump(strip(results), open(args.out / "comparison.json", "w"), indent=2)

    print("\nBudget (top-N crops per room): recall@5 / recall@1 / crop hit rate")
    for n in BUDGETS:
        line = f"N={n:>2} "
        for v in variants:
            r = results["budget"][v][n]
            line += f"| {v:18} {r['recall@5']:.3f} / {r['recall@1']:.3f} / {r['crop_hit_rate']:.3f} ({r['crops_per_room']:.1f}) "
        print(line)
    print("\nPaired diff Golden - YOLO-World(agnostic), recall@5 [95% CI]:")
    for n, (d, lo, hi) in results["paired_diff_recall@5"]["yoloworld_agnostic"].items():
        print(f"  N={n:>2}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\nConfidence sweep: recall@5 (crops/room)")
    for c in CONFS:
        print(f"conf {c:.2f} " + " ".join(f"| {v:18} {results['sweep'][v][c]['recall@5']:.3f} ({results['sweep'][v][c]['crops_per_room']:.1f})" for v in variants))
    print("\nlatency (median ms):", results["latency_ms_median"], device)


if __name__ == "__main__":
    main()
