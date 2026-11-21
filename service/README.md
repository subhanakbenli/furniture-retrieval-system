# Retrieval service

FastAPI + Qdrant + Streamlit. A fine-tuned YOLO11m detector (`weights/Golden.pt`, 5 classes: Furniture, Textile, Lighting, Decoration, Appliance) finds objects in a room photo; each crop is embedded with CLIP (`clip-ViT-B-32`) and matched against a product catalog in Qdrant.

## Run with Docker (recommended)

```bash
docker compose up -d --build
```

| Container | Port | What |
|---|---|---|
| `ui` | 8501 | Streamlit web UI |
| `api` | 8000 | FastAPI (Swagger at `/docs`) |
| `qdrant` | 6333 (localhost only) | vector database |

The image is CPU-only (PyTorch CPU wheels, CLIP baked in at build time). Catalog vectors and uploaded product images persist in the `qdrant_storage` and `uploads` volumes; `docker compose down -v` deletes them.

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `MODEL_PATH` | `weights/Golden.pt` | api |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | api |
| `API_URL` | `http://localhost:8000` | ui |

## Try the model

```bash
# detections only, as JSON
curl -F "file=@room.jpg" -F "conf_threshold=0.25" http://localhost:8000/detect

# image with post-processed boxes drawn
curl -F "file=@room.jpg" http://localhost:8000/visualize -o out.jpg

# a whole folder -> results/predictions.jsonl (+ annotated images)
pip install requests pillow
python scripts/predict_folder.py path/to/images --out results/ --save-images
```

`predictions.jsonl` has one line per image:

```json
{"filename": "room.jpg", "image": {"width": 1684, "height": 1263},
 "detections": [{"label": "Furniture", "confidence": 0.968, "box": [1214, 2, 1614, 1040]}]}
```

Boxes are `[x1, y1, x2, y2]` in pixels, after post-processing (see [docs/postprocess.md](docs/postprocess.md)).

## Run without Docker

Python 3.10+:

```bash
docker compose up -d qdrant
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements-docker.txt
uvicorn src.main:app --port 8000
streamlit run src/frontend.py      # second terminal
```

`requirements.txt` holds the exact environment of the original GPU deployment (Linux, CUDA 12.8).

## Layout

- `src/main.py` – API endpoints
- `src/ml_engine.py` – YOLO detection + CLIP embedding
- `src/postprocess.py` – box merging / conflict rules (tests: `verify_postprocess.py`)
- `src/database.py` – Qdrant access
- `src/frontend.py` – Streamlit UI
- `scripts/predict_folder.py` – batch detection over a folder
- `docs/` – API, architecture and post-processing notes
