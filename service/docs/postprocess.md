# Object Detection Post-Processing

This document describes the advanced post-processing logic designed to refine the raw object detection results from the YOLO model. This logic helps in reducing false positives, resolving ambiguities between overlapping classes, and merging fragmented detections of large objects.

## 1. Core Concepts

### Intersection over Foreground (IoF)
Unlike the standard Intersection over Union (IoU), IoF is calculated relative to the area of the *smaller* box (or the "foreground" candidate).

$$ IoF = \frac{\text{Area of Intersection}}{\text{Area of Smaller Box}} $$

*   **Purpose**: To detect when one object is largely "inside" another or when a smaller fragment overlaps significantly with a larger part, even if their total Union area is large (which would lower the IoU).

## 2. Post-Processing Rules

The `PostProcessor` applies the following rules sequentially to the raw detections:

### Rule 1: Cross-Class Conflict Resolution (Ambiguity Filter)
*   **Condition**: Two bounding boxes have **different class labels** but an extremely high overlap (**IoU > 0.97**).
*   **Action**: **Delete BOTH boxes**.
*   **Rationale**: When the model detects the exact same region as two different classes with near-perfect overlap, it indicates high uncertainty and ambiguity (e.g., classifying the same object as both "Furniture" and "Decoration"). Removing both prevents false positives.

### Rule 2: Same-Class Merge (Fragment Reassembly)
*   **Condition**:
    1.  Two bounding boxes belong to the **same class**.
    2.  The smaller box is largely inside the larger one (**IoF > `iof_threshold`**, default **0.5**; see Section 3).
*   **Action**:
    1.  **Merge** the two boxes into a single new box that covers the extent of both (Union).
    2.  The "Parent" (larger) box is updated to these new coordinates.
    3.  The "Child" (smaller) box is removed.
*   **Rationale**: Large objects (like sofas or cabinets) are sometimes detected as multiple separate parts by YOLO. This rule "glues" these parts back together into a single, comprehensive bounding box, ensuring the downstream search engine gets a complete view of the object.

---

## 3. Dynamic Thresholds
The post-processing logic is not static; it is controlled by dynamic parameters exposed via the API and Frontend:

*   **Confidence Threshold (`conf_threshold`)**: Controls the raw YOLO detection sensitivity. Lower values find more objects but increase noise.
*   **IoF Threshold (`iof_threshold`)**: Controls the **Same-Class Merge** rule.
    *   **Higher values (> 0.8)**: Only merge if the smaller box is almost entirely inside the larger one.
    *   **Lower values (< 0.5)**: Aggressively merge overlapping boxes.
    *   *Default*: 0.5

## 4. Integration Strategy
The PostProcessor is fully integrated into the `MLEngine`.

1.  **Detections**: The `detect_objects` method receives `conf_threshold` and `iof_threshold` from the API request.
2.  **Application**: These values are passed directly to `PostProcessor.apply_rules`.
3.  **Result**: The downstream cropping and embedding generation are performed strictly on the *processed* results, ensuring the search engine only sees "clean" objects.
