# Training

YOLO11m detector for 5 superclasses (Furniture, Textile, Lighting, Decoration, Appliance), trained on YOLO-World pseudo-labels with Ultralytics. Full write-up: [../docs/02-training-and-evaluation.md](../docs/02-training-and-evaluation.md).

| Script | What it does |
|---|---|
| `train_model.py` | Trains or fine-tunes a YOLO model from a `data.yaml` |
| `mine_hard_examples.py` | Runs a model over an image pool and samples hard examples: weak detections (confidence 0.15–0.55, 60%), context confusion (20%) and negatives (20%) |
| `prepare_hard_examples.py` | Builds a YOLO dataset from the mined CSV. Writes the same images to `train` and `val`, so stage-1 validation metrics are inflated |
| `evaluate_model.py` | Batch inference on a test folder with per-class detection summary and visualizations |
| `inference_with_postprocessing.py` | Earlier, training-side version of the post-processing rules (IoF default 0.85). The service uses `service/src/postprocess.py` (default 0.5) |
| `config_utilities.py` | Shared paths, class names and helpers |

## Runs

Run order (each phase starts from the previous phase's `best.pt`):

1. `runs/yolov11m_superclass`: Gen-2 base, `yolo11m.pt` on the 40k golden set, 50 epochs
2. `runs/stage1_adaptation_hard10k`: 10k mined hard examples
3. `runs/stage2_consolidation_44k`: 44k consolidated set (stopped after 11 of 40 epochs, best at epoch 1)
4. `runs/phase3_lighting_specialization`: Lighting-focused set, AdamW lr 1e-4
5. `runs/phase3_5_rebalance` and `runs/phase3_5_rebalance2`: back to the 44k set to recover the other classes. The second run uses an explicit AdamW lr 1e-4, because `optimizer=auto` had overridden `lr0`. Its `best.pt` is the final `Golden.pt`

Datasets and intermediate checkpoints are not included; the final model is `../service/weights/Golden.pt`. Each run folder has the exact `args.yaml`, per-epoch `results.csv`, PR/F1 curves and confusion matrices.
