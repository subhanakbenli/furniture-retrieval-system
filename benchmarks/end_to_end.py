"""
End-to-end pipeline benchmark: detector -> crops -> CLIP embeddings -> catalog search,
timed per stage, for Golden.pt and a YOLO-World baseline on the synthetic-room set.

Both pipelines share everything except the detector: same CLIP model (batched crop
embedding), same catalog search (cosine, top-K per crop within the detected category),
same min crop area and similarity threshold. Golden.pt runs with the service's
post-processing; YOLO-World runs with the 114 labelling prompts mapped to the 5
superclasses, optionally with class-agnostic NMS.

Reported per detector: retrieval quality (room recall@1/@5, precision@5), crops and
CLIP calls per room, and latency per stage (mean / median / p90) plus total throughput.
The first rooms are processed once as warm-up and excluded from timing. Pipelines are
interleaved room by room in rotating order, so load changes on the machine affect all
of them equally.

Usage:
    python benchmarks/end_to_end.py --data path/to/dataset --out results/end_to_end
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_benchmark import CLIP_MODEL_ID, REPO, YOLO_WORLD_HIERARCHY, load_catalog  # noqa: E402
from src.postprocess import PostProcessor  # noqa: E402

K, SIM_THRESHOLD, MIN_AREA, CONF, NMS_IOU = 5, 0.2, 800, 0.25, 0.6


class GoldenDetector:
    name = "Golden.pt (YOLO11m) + post-processing"

    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(str(REPO / "service/weights/Golden.pt"))
        self.post = PostProcessor(iof_threshold=0.5)

    def __call__(self, image):
        r = self.model(image, conf=CONF, iou=NMS_IOU, verbose=False)[0]
        b, s, c = r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()
        b, s, c = self.post.apply_rules(b, s, c)
        return b, [self.model.names[x] for x in c]


class YoloWorldDetector:
    def __init__(self, agnostic=False, model=None):
        from ultralytics import YOLOWorld
        self.agnostic = agnostic
        self.name = "YOLO-World v8l (114 prompts)" + (" + agnostic NMS" if agnostic else "")
        prompts, self.supers = [], []
        for sup, items in YOLO_WORLD_HIERARCHY.items():
            prompts += items
            self.supers += [sup] * len(items)
        if model is None:
            model = YOLOWorld("yolov8l-worldv2.pt")
            model.set_classes(prompts)
        self.model = model

    def __call__(self, image):
        r = self.model(image, conf=CONF, iou=NMS_IOU, verbose=False)[0]
        boxes, scores, cls = r.boxes.xyxy, r.boxes.conf, r.boxes.cls.int()
        if self.agnostic and len(boxes):
            keep = torchvision.ops.nms(boxes, scores, NMS_IOU)
            boxes, cls = boxes[keep], cls[keep]
        return boxes.tolist(), [self.supers[x] for x in cls.tolist()]


def process_room(detector, image, gt, clip, proc, catalog, device):
    """Run the full pipeline on one room; return stage timings (ms) and quality."""
    t0 = time.perf_counter()
    boxes, labels = detector(image)
    t1 = time.perf_counter()

    crops, cats = [], []
    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if (x2 - x1) * (y2 - y1) >= MIN_AREA and lab in catalog:
            crops.append(image.crop((x1, y1, x2, y2)))
            cats.append(lab)
    feats = np.zeros((0, 512), dtype=np.float32)
    if crops:
        with torch.no_grad():
            px = proc(images=crops, return_tensors="pt")["pixel_values"].to(device)
            f = clip.visual_projection(clip.vision_model(pixel_values=px).pooler_output)
            feats = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
    t2 = time.perf_counter()

    found1, found5 = set(), set()
    for f, cat in zip(feats, cats):
        embs, ids = catalog[cat]
        sims = embs @ f
        order = [j for j in np.argsort(-sims)[:K] if sims[j] >= SIM_THRESHOLD]
        if order:
            found1.add(int(ids[order[0]]))
        found5.update(int(ids[j]) for j in order)
    t3 = time.perf_counter()

    timing = {"detect": t1 - t0, "embed": t2 - t1, "search": t3 - t2, "total": t3 - t0}
    return {k: v * 1000 for k, v in timing.items()}, {
        "r1": len(found1 & gt) / len(gt),
        "r5": len(found5 & gt) / len(gt),
        "p5": len(found5 & gt) / len(found5) if found5 else 0.0,
        "crops": len(crops),
    }


def run_interleaved(detectors, clip, proc, catalog, rooms, data_dir, warmup, device):
    images = {r["room_id"]: Image.open(data_dir / "room_final" / Path(r["image_path"]).name).convert("RGB") for r in rooms}
    timings = [{"detect": [], "embed": [], "search": [], "total": []} for _ in detectors]
    quality = [{"r1": [], "r5": [], "p5": [], "crops": []} for _ in detectors]

    for i, room in enumerate(rooms[:warmup] + rooms):
        image, gt = images[room["room_id"]], set(room["object_ids"])
        order = [(i + s) % len(detectors) for s in range(len(detectors))]  # rotate who goes first
        for d in order:
            t, q = process_room(detectors[d], image, gt, clip, proc, catalog, device)
            if i < warmup:
                continue
            for k, v in t.items():
                timings[d][k].append(v)
            for k, v in q.items():
                quality[d][k].append(v)
        if i >= warmup and (i - warmup + 1) % 50 == 0:
            print(f"{i - warmup + 1}/{len(rooms)} rooms")

    stats = lambda v: {"mean": float(np.mean(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
    return [{
        "detector": det.name,
        "rooms": len(q["r5"]),
        "recall@1": float(np.mean(q["r1"])),
        "recall@5": float(np.mean(q["r5"])),
        "precision@5": float(np.mean(q["p5"])),
        "crops_per_room": float(np.mean(q["crops"])),
        "latency_ms": {k: stats(v) for k, v in t.items()},
        "throughput_rooms_per_s": 1000 / float(np.mean(t["total"])),
        "_r5": q["r5"],
    } for det, t, q in zip(detectors, timings, quality)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/end_to_end"))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog, _ = load_catalog(args.data / "dataset.json")
    rooms = json.load(open(args.data / "room_final" / "rooms.json"))[: args.limit]
    clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    world = YoloWorldDetector()
    detectors = [GoldenDetector(), world, YoloWorldDetector(agnostic=True, model=world.model)]
    results = run_interleaved(detectors, clip, proc, catalog, rooms, args.data, args.warmup, device)
    for res in results:
        lat = res["latency_ms"]
        print(f"{res['detector']}: recall@5 {res['recall@5']:.3f}, recall@1 {res['recall@1']:.3f}, "
              f"crops {res['crops_per_room']:.1f}, total {lat['total']['mean']:.0f} ms "
              f"(detect {lat['detect']['mean']:.0f} / embed {lat['embed']['mean']:.0f} / search {lat['search']['mean']:.1f})")

    # paired bootstrap on recall@5, Golden minus each baseline
    rng = np.random.default_rng(0)
    g = np.asarray(results[0]["_r5"])
    for res in results[1:]:
        b = np.asarray(res["_r5"])
        idx = rng.integers(0, len(g), size=(10000, len(g)))
        d = (g[idx] - b[idx]).mean(axis=1)
        res["golden_minus_this_recall@5"] = [float((g - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    for res in results:
        res.pop("_r5")

    summary = {"device": device, "torch_threads": torch.get_num_threads(),
               "settings": {"conf": CONF, "nms_iou": NMS_IOU, "min_area": MIN_AREA, "k": K,
                            "similarity_threshold": SIM_THRESHOLD, "clip": CLIP_MODEL_ID, "warmup_rooms": args.warmup},
               "results": results}
    json.dump(summary, open(args.out / "end_to_end.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
