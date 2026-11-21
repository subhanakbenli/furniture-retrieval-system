# Benchmarks

Retrieval benchmark on **250 synthetic rooms**. Each room image was generated from 2–5 products of a 190-item catalog, so the products in every room are known. The benchmark compares the full pipeline (detect → crop → CLIP → catalog search) with **Golden.pt** against an open-vocabulary **YOLO-World v8l** baseline that uses the same 114 prompts we used for labelling, mapped to the same 5 classes.

Everything except the detector is shared between the two pipelines: CLIP ViT-B/32 crop embeddings, cosine top-5 search within the detected category, a 0.2 similarity threshold and an 800 px minimum crop area. All runs were on CPU (Apple M5, Docker, 8 threads).

## End-to-end pipeline

`end_to_end.py` times every stage per room. The pipelines run interleaved, room by room in rotating order, and 5 warm-up rooms are excluded.

| Pipeline | Recall@5 | Recall@1 | Crops / room | Detect | Embed | **Total (mean / median)** | Rooms / s |
|---|---|---|---|---|---|---|---|
| **Golden.pt (YOLO11m) + post-processing** | 0.685 | 0.431 | **16.6** | **266 ms** | **376 ms** | **642 / 596 ms** | **1.56** |
| YOLO-World v8l | 0.729 | 0.476 | 24.2 | 417 ms | 543 ms | 960 / 937 ms | 1.04 |
| YOLO-World v8l + class-agnostic NMS | 0.718 | 0.466 | 19.0 | 424 ms | 430 ms | 854 / 810 ms | 1.17 |

- **1.5× faster end to end**, 1.57× faster detection.
- **31% fewer crops**, so 31% fewer CLIP calls. Embedding is the most expensive stage, so the detector's output size matters as much as its own speed.
- **94% of the baseline's room recall** (0.685 vs 0.729) at two thirds of the cost.

## Equal crop budget

Room-level recall grows with the number of crops, because every crop retrieves 5 products. `detector_comparison.py` removes this effect by keeping each detector's N most confident crops per room. The table shows recall@5; the difference column has 95% CIs from a paired bootstrap over rooms.

| Crops / room | Golden.pt | YOLO-World | YOLO-World + agnostic NMS | Golden − YOLO-World |
|---|---|---|---|---|
| 3 | **0.301** | 0.258 | 0.287 | +0.043 [+0.007, +0.079] |
| 5 | **0.441** | 0.358 | 0.404 | **+0.084 [+0.045, +0.122]** |
| 8 | **0.565** | 0.490 | 0.516 | **+0.075 [+0.041, +0.110]** |
| 10 | **0.608** | 0.539 | 0.567 | **+0.069 [+0.036, +0.101]** |
| 15 | 0.663 | 0.625 | **0.673** | +0.038 [+0.009, +0.068] |

At a fixed crop budget, Golden.pt retrieves more of the room's products: **+7–8 points recall@5 at 5–10 crops per room**. Against the agnostic-NMS variant, its strongest form, the lead is +4–5 points at 8–10 crops. Its most confident boxes are the objects that matter for retrieval. Without a budget, YOLO-World reaches higher recall by emitting more boxes (up to 30–40 per room at low confidence).

## Run it

The dataset is not distributed. With a copy at `path/to/dataset` (`dataset.json` + `room_final/`):

```bash
cd service && docker compose build
docker compose run --rm --no-deps -v $PWD/..:/repo -v path/to/dataset:/data:ro -w /tmp api \
  sh -c "pip install -q openai-clip && python /repo/benchmarks/end_to_end.py --data /data --out /repo/benchmarks/results/e2e"
```

| Script | What it measures |
|---|---|
| [`end_to_end.py`](end_to_end.py) | Full pipeline: retrieval quality and per-stage latency for both detectors |
| [`detector_comparison.py`](detector_comparison.py) | Equal crop budgets, confidence sweep, crop hit rate, paired bootstrap |
| [`retrieval_benchmark.py`](retrieval_benchmark.py) | Room recall/precision at K = 1, 3, 5, 10 for one detector, with a random-retrieval control |

`openai-clip` is only needed for YOLO-World's text encoder. Results are written to `benchmarks/results/` (git-ignored).
