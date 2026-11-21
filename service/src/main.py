from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager
from PIL import Image
import io
import json
from typing import List, Dict, Any

from src.ml_engine import ml_engine
from src.database import QdrantHandler
from fastapi.staticfiles import StaticFiles
import uuid
import traceback
from fastapi.responses import StreamingResponse
from PIL import ImageDraw, ImageFont

# Initialize services
qdrant_handler = QdrantHandler()




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("Startup: Ensuring models are loaded...")
    _ = ml_engine
    yield
    print("Shutdown: Cleaning up...")

app = FastAPI(title="Object Search API", lifespan=lifespan)
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iof_threshold: float = Form(0.5)
):
    """
    Detection only (no catalog search): post-processed boxes as JSON.
    Useful for running the model over a test set.
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        detections = ml_engine.detect_objects(image, conf_threshold=conf_threshold, iof_threshold=iof_threshold)

        return {
            "filename": file.filename,
            "image": {"width": image.width, "height": image.height},
            "detections": [
                {"label": d["label"], "confidence": d["confidence"], "box": d["box"]}
                for d in detections
            ]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store")
async def store_image(
    files: List[UploadFile] = File(...),
    metadata: str = Form(...)
):
    """
    Store one or more images for a SINGLE object.
    """
    try:
        try:
            metadata_json = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON metadata")

        # Generate a shared object_id if not present
        if "object_id" not in metadata_json:
            metadata_json["object_id"] = str(uuid.uuid4())
            
        object_id = metadata_json["object_id"]
        stored_points = []
        
        for file in files:
            # Read image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Get Embedding
            embedding = ml_engine.get_embedding(image)
            
            # Save exact image to disk
            point_id = str(uuid.uuid4())
            filename = f"{point_id}.jpg"
            image.save(f"static/uploads/{filename}")
            
            # Per-point metadata includes the shared object_id and specific image_url
            point_meta = metadata_json.copy()
            point_meta["image_url"] = f"/static/uploads/{filename}"
            
            # Store in Qdrant
            qdrant_handler.store_object(embedding, point_meta, point_id=point_id)
            stored_points.append(point_id)
        
        return {
            "status": "success",
            "object_id": object_id,
            "point_ids": stored_points,
            "metadata": metadata_json
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-and-search")
async def detect_and_search(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iof_threshold: float = Form(0.5),
    search_threshold: float = Form(0.60)
):
    """
    1. Detect objects in the uploaded scene.
    2. For each detection, crop and get embedding.
    3. Search in Qdrant for similar objects (Grouped).
    """
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Detect objects
        detections = ml_engine.detect_objects(image, conf_threshold=conf_threshold, iof_threshold=iof_threshold)
        
        results = []
        for det in detections:
            # Get embedding for the crop
            crop_emb = ml_engine.get_embedding(det['crop'])
            
            # Search in DB (GROUPED) - Filter by detected Class Name
            similar_objects = qdrant_handler.search_similar_groups(
                crop_emb, 
                limit=3, 
                filter_category=det['label']
            )
            
            # Filter by search threshold
            filtered_matches = [m for m in similar_objects if m['score'] >= search_threshold]
            
            # Format result (removing PIL image for JSON serialization)
            results.append({
                "label": det['label'],
                "confidence": det['confidence'],
                "box": det['box'],
                "matches": filtered_matches
            })
            
        return {"detections_count": len(results), "results": results}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def visualize_detections(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iof_threshold: float = Form(0.5)
):
    """
    Returns the image with bounding boxes drawn (for visualizing post-processing).
    """
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Detect
        detections = ml_engine.detect_objects(image, conf_threshold=conf_threshold, iof_threshold=iof_threshold)
        
        # Draw
        draw = ImageDraw.Draw(image)
        try:
            # Attempt to use a larger font
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = None

        for det in detections:
            box = det['box']
            label = f"{det['label']} {det['confidence']:.2f}"
            
            draw.rectangle(box, outline="red", width=3)
            
            # Label
            text_loc = (box[0], box[1] - 20 if box[1] > 20 else box[1])
            if font:
                left, top, right, bottom = draw.textbbox(text_loc, label, font=font)
                draw.rectangle((left, top, right, bottom), fill="red")
                draw.text(text_loc, label, fill="white", font=font)
            else:
                draw.rectangle([text_loc, (text_loc[0] + 100, text_loc[1] + 10)], fill="red")
                draw.text(text_loc, label, fill="white")

        # Return image
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/objects")
def list_objects(limit: int = 10, offset: str = None):
    """
    List objects with pagination.
    Offset is the point_id to start after (from Qdrant scroll).
    """
    try:
        results, next_offset = qdrant_handler.list_objects(limit=limit, offset=offset)
        return {"results": results, "next_offset": next_offset}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/objects/{point_id}")
def delete_object(point_id: str):
    """Delete an object and its image file."""
    try:
        # We might want to get the image path first to delete it (optional optimization)
        # For now, just try to delete the file if it exists
        filename = f"static/uploads/{point_id}.jpg"
        if os.path.exists(filename):
            os.remove(filename)
            
        qdrant_handler.delete_object(point_id)
        return {"status": "success", "deleted_id": point_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/objects/{point_id}")
async def update_object(
    point_id: str,
    metadata: str = Form(...),
    files: List[UploadFile] = File(None)
):
    """
    Update object.
    1. Updates metadata for the ENTIRE object group (all images of this object).
    2. If files provided, adds them as NEW variants (does not replace the current one).
    """
    try:
        try:
            metadata_json = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON metadata")
        
        # 1. Identify Group ID
        # We need to find the object_id of the point being edited.
        # Ideally, it's passed in metadata, but safe to fetch or trust metadata['object_id']
        object_id = metadata_json.get("object_id")
        
        if not object_id:
            # Fallback: Treat point_id as object_id or just update this point?
            # For robustness, let's assume valid data has object_id.
            # If missing, maybe we should fetch the point first?
            # Let's rely on frontend passing it back.
            object_id = str(uuid.uuid4()) # Should not happen if data is correct
            metadata_json["object_id"] = object_id

        # image_url is per point (one per variant image); never copy it across the group
        metadata_json.pop("image_url", None)

        # 2. Update Metadata for ALL points in this group
        qdrant_handler.update_group_payload(object_id=object_id, metadata=metadata_json)
        
        # 3. Add New Images (if any)
        if files:
            for file in files:
                # Read image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Get Embedding
                embedding = ml_engine.get_embedding(image)
                
                # Save
                new_point_id = str(uuid.uuid4())
                filename = f"{new_point_id}.jpg"
                image.save(f"static/uploads/{filename}")
                
                # Metadata
                new_meta = metadata_json.copy()
                new_meta["image_url"] = f"/static/uploads/{filename}"
                
                # Store
                qdrant_handler.store_object(embedding, new_meta, point_id=new_point_id)
        
        return {"status": "success", "id": point_id, "object_id": object_id}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
