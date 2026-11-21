# API Reference

This document details the REST API endpoints available in the retrieval service.

## Base URL
All endpoints are relative to the base URL (default: `http://localhost:8000`).

---

## 1. Store Object
Add a new object (or new images for an existing object) to the database.

**Endpoint:** `POST /store`

### Request
*   **Content-Type**: `multipart/form-data`
*   **Parameters**:
    *   `files` (required): One or more image files (UploadFile).
    *   `metadata` (required): A JSON string containing object details.
        *   `name` (string): Object name.
        *   `category` (string): Object category.
        *   `description` (string): Optional description.
        *   `price` (string): Optional price.
        *   `object_id` (string, optional): If provided, adds images to this existing object group. If valid UUID is omitted, a new one is generated.

### Example (Python requests)
```python
import requests
import json

url = "http://localhost:8000/store"

metadata = {
    "name": "Eames Chair",
    "category": "Furniture",
    "description": "Mid-century modern leather chair",
    "price": "3500"
}

files = [
    ("files", ("chair_front.jpg", open("chair_front.jpg", "rb"), "image/jpeg")),
    ("files", ("chair_side.jpg", open("chair_side.jpg", "rb"), "image/jpeg"))
]

response = requests.post(url, files=files, data={"metadata": json.dumps(metadata)})
print(response.json())
```

### Response
```json
{
  "status": "success",
  "object_id": "550e8400-e29b-41d4-a716-446655440000",
  "point_ids": ["uuid-1", "uuid-2"],
  "metadata": { ... }
}
```

---

## 2. Detect and Search
Detect objects in a scene and find similar items in the database.

**Endpoint:** `POST /detect-and-search`

### Process Overview
1.  **Detect**: Runs the fine-tuned YOLO11m detector (`weights/Golden.pt`) on the uploaded image.
2.  **Filter**: For each detection, limits the search space to objects in the database with the **same category** (matched by label).
3.  **Search**: Finds visually similar items within that category using CLIP embeddings.
4.  **Match**: Returns the top matches grouped by Object ID.

### Request
*   **Content-Type**: `multipart/form-data`
*   **Parameters**:
    *   `file` (required): A single image file (the scene to analyze).
    *   `conf_threshold` (float, optional, default=0.25): YOLO detection confidence.
    *   `iof_threshold` (float, optional, default=0.5): Merge/IOF threshold.
    *   `search_threshold` (float, optional, default=0.60): Minimum similarity score for DB matches.

### Example
```python
files = {"file": ("living_room.jpg", open("living_room.jpg", "rb"), "image/jpeg")}
response = requests.post("http://localhost:8000/detect-and-search", files=files)
```

### Response
```json
{
  "detections_count": 2,
  "results": [
    {
      "label": "chair",
      "confidence": 0.85,
      "box": [100, 200, 300, 400],
      "matches": [
        {
          "id": "point-uuid",
          "score": 0.92,
          "metadata": {
            "name": "Eames Chair",
            "image_url": "/static/uploads/..."
          },
          "group_id": "object-uuid"
        }
      ]
    }
  ]
}
```

---

## 3. Manage Objects

### 3.1 List Objects
Retrieve stored objects with pagination.

**Endpoint:** `GET /objects`

*   **Query Parameters**:
    *   `limit` (int, default=10): Number of items to return.
    *   `offset` (string, optional): The `next_offset` from the previous response (for pagination).

### 3.2 Update Object
Update metadata or add new image variants to an object.

**Endpoint:** `PUT /objects/{point_id}`

*   **Content-Type**: `multipart/form-data`
*   **Parameters**:
    *   `metadata` (required): JSON string with updated fields.
    *   `files` (optional): New images to append to this object.

### 3.3 Delete Object
Remove a specific image point and its record.

**Endpoint:** `DELETE /objects/{point_id}`

*   **Path Parameters**:
    *   `point_id`: The UUID of the specific image/point to delete.

---

## 4. Visualize Detection (Debug)
Visual debugging endpoint to see exactly what the model counts as a valid object after post-processing. Useful for tuning thresholds.

**Endpoint:** `POST /visualize`

*   **Content-Type**: `multipart/form-data`
*   **Parameters**:
    *   `file` (required): Image file.
    *   `conf_threshold` (float, optional, default=0.25): YOLO confidence threshold.
    *   `iof_threshold` (float, optional, default=0.5): Intersection over Foreground threshold for merging fragmented boxes.

### Response
Returns a `image/jpeg` stream of the input image with bounding boxes drawn (red).

---

## 5. Detect Only

Runs the detector and post-processing without searching the catalog. Use it to evaluate the model on a test set (see `scripts/predict_folder.py`).

*   **URL**: `/detect`
*   **Method**: `POST`
*   **Content-Type**: `multipart/form-data`

### Request
*   `file` (file, required): Scene image.
*   `conf_threshold` (float, optional, default=0.25): YOLO confidence threshold.
*   `iof_threshold` (float, optional, default=0.5): IoF threshold for merging same-class boxes.

### Example
```bash
curl -F "file=@room.jpg" http://localhost:8000/detect
```

### Response
```json
{
  "filename": "room.jpg",
  "image": {"width": 1684, "height": 1263},
  "detections": [
    {"label": "Furniture", "confidence": 0.968, "box": [1214, 2, 1614, 1040]}
  ]
}
```
