FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# OpenCV (pulled in by ultralytics) needs these shared libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1 \
    && pip install -r requirements-docker.txt

# Bake the CLIP model into the image so containers start without downloading it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"

COPY weights ./weights
COPY src ./src

EXPOSE 8000 8501
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
