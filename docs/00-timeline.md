# Project timeline

The project's phases from November 2025 to February 2026, with the main technical decision in each one and the reason for it.

| Phase | Dates | Outcome |
|---|---|---|
| 1. Problem definition and public datasets | Nov 2025 | Merged public datasets, YOLO11-seg prototype |
| 2. Synthetic data and SAM3 labeling | late Nov – early Dec 2025 | SAM3 used as a mask "teacher" on a small pool |
| 3. Large image pool and golden set | Dec 2025 | 1.85M images, 40,000-image golden set |
| 4. YOLO-World labels and active learning | late Dec 2025 | Final YOLO11m detector (`Golden.pt`) |
| 5. Retrieval service and evaluation | Jan – Feb 2026 | FastAPI + Qdrant + Streamlit service, retrieval benchmark |

## 1. Problem definition and public datasets (Nov 2025)

- **Approach:** detect objects, crop them, embed the crops and match them against a product catalog. This is the same structure as DeepFurniture ([arXiv 1911.09299](https://arxiv.org/abs/1911.09299)).
- **Decision: build our own dataset.** The first model was a YOLO11 segmentation model trained on about 700 images merged from several public datasets. Each source dataset labeled only its own classes, so objects such as lamps in a sofa dataset were left unlabeled. The model then learned to treat them as background. A larger model did not fix this, which showed that the data was the problem, not model capacity.
- **Decision: use off-the-shelf CLIP for embeddings.** Metric learning was put off until retrieval quality showed a need for it.

## 2. Synthetic data and SAM3 labeling (late Nov – early Dec 2025)

- BlenderProc synthetic rooms and a 14k augmented dataset were tried, and neither improved results on real photos.
- **Decision: auto-label real images with a foundation model.** SAM3 was prompted class by class to produce masks for a few room categories ([`sam_annotation_living_room.ipynb`](../notebooks/executed/sam_annotation_living_room.ipynb)). A YOLO11-seg model trained on these labels passed 0.60 mAP50.

## 3. Large image pool and golden set (Dec 2025)

- **Data:** 1,912,541 image links collected from public interior-design websites across 42 room categories, and 1,850,924 images downloaded (the download was split across three machines).
- **Decision: 5 superclasses plus 114 fine-grained prompts.** The prompts map to Furniture, Textile, Lighting, Decoration and Appliance. The detector learns 5 balanced classes, while the labeler still gets specific text prompts.
- **Decision: select a diverse subset instead of labeling everything.** Images were embedded with CLIP ViT-B/32, near-duplicates were removed with FAISS at cosine 0.98 (leaving 1,713,762), and 40,000 images were selected: 28,000 by K-Means and 12,000 by Farthest Point Sampling ([`golden_dataset_creator.ipynb`](../notebooks/executed/golden_dataset_creator.ipynb), [`../data_pipeline/dataset_creation/`](../data_pipeline/dataset_creation/)).

## 4. YOLO-World labels and active learning (late Dec 2025)

- **Decision: SAM3 → YOLO-World.** SAM3 took about 2.3 s per image on an A100. With 114 prompts, labeling the golden set was estimated at about 90 hours ([`Golden_Annotation.ipynb`](../notebooks/executed/Golden_Annotation.ipynb)). YOLO-World (yolov8l-worldv2) labeled the 40k set in about 15 minutes. It later labeled the full 1.85M pool in 11 h 28 min at 44.8 images/s ([`yolo_world_full_pool_annotation.ipynb`](../annotation/yolo_world_full_pool_annotation.ipynb)).
- **Decision: segmentation → bounding boxes.** Retrieval only needs a crop of each object, so masks add labeling cost without improving the result.
- **Baseline:** YOLO11m fine-tuned on the golden set for 50 epochs reached 0.695 mAP50 and 0.613 mAP50-95 ([`yolov11m_superclass`](../training/runs/yolov11m_superclass/)). Lighting was the weakest class.
- **Decision: active learning for weak classes.** Training ran in four phases: 10k mined hard examples, a 44k consolidation set, Lighting specialization, and a rebalancing pass ([`mine_hard_examples.py`](../training/mine_hard_examples.py), [`../training/`](../training/)). The final model reached P 0.806, R 0.637, mAP50 0.735 and mAP50-95 0.649. Details are in [02-training-and-evaluation.md](02-training-and-evaluation.md).

## 5. Retrieval service and evaluation (Jan – Feb 2026)

- **Service:** FastAPI + Qdrant with a Streamlit UI for search, adding products and managing inventory ([`../service/`](../service/)). It ran as a demo on an AWS EC2 GPU instance.
- **Decision: post-process detections.** Fragmented or duplicate boxes for the same object produced redundant queries. Same-class boxes that overlap heavily are merged, and near-identical boxes with conflicting classes are dropped.
- **Decision: evaluate with synthetic rooms.** A retrieval benchmark needs ground truth, so room images were generated from catalog products, with each room labeled by the product IDs it contains. On 250 rooms the Golden.pt pipeline reaches 94% of the YOLO-World pipeline's room recall at 1.5× its speed, with higher recall at equal crop budgets ([benchmarks](../benchmarks/)).
- **Feb 2026:** final code and documentation published.
