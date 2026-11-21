import numpy as np
from typing import List, Tuple, Dict, Any

class PostProcessor:
    """
    Handles post-processing of object detection results.
    
    Rules:
    1. Cross-Class Conflict Resolution: If two boxes have different classes but high IoU (>0.97), delete BOTH.
    2. Same-Class Merge: If two boxes have the same class and one is largely inside the other (IoF > threshold),
       merge them into a single larger box.
    """

    def __init__(self, iof_threshold: float = 0.5):
        self.iof_threshold = iof_threshold
        # Mapping for logging purposes (optional, can be passed in)
        self.processing_log = []

    def calculate_iof(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Foreground (Area of Intersection / Area of Smaller Box).
        Used to detect containment.
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Area of box1 (foreground candidate)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        if box1_area == 0: return 0
        return inter_area / box1_area

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Areas
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union
        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0
        
        return inter_area / union_area

    def apply_rules(self, boxes: List[List[float]], scores: List[float], class_ids: List[int], iof_threshold: float = None) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Apply both conflict resolution and merge rules.
        """
        if iof_threshold is not None:
             self.iof_threshold = iof_threshold

        n_boxes = len(boxes)
        if n_boxes == 0: return [], [], []

        # Convert to mutable lists if they aren't already
        boxes = [list(b) for b in boxes]
        
        keep = [True] * n_boxes
        self.processing_log = []
        
        # --- RULE 1: Cross-Class Conflict Resolution (IoU > 0.97) ---
        for i in range(n_boxes):
            if not keep[i]: continue
            for j in range(i + 1, n_boxes):
                if not keep[j]: continue
                
                # Different classes?
                if class_ids[i] != class_ids[j]:
                    iou = self.calculate_iou(boxes[i], boxes[j])
                    
                    if iou > 0.97:
                        keep[i] = False
                        keep[j] = False
                        self.processing_log.append({
                            'rule': 'conflict_resolution',
                            'action': 'delete_both',
                            'details': f"Ids {class_ids[i]} & {class_ids[j]} overlap {iou:.3f}"
                        })
        
        # --- RULE 2: Same-Class Merge (Union) ---
        # Sort by area (Largest -> Smallest) to absorb small into large
        areas = [(boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]) for i in range(n_boxes)]
        sorted_indices = sorted(range(n_boxes), key=lambda k: areas[k], reverse=True)

        for idx_i in range(len(sorted_indices)):
            i = sorted_indices[idx_i]
            if not keep[i]: continue

            for idx_j in range(idx_i + 1, len(sorted_indices)):
                j = sorted_indices[idx_j]
                if not keep[j]: continue
                
                # Must be same class
                if class_ids[i] != class_ids[j]:
                    continue

                # Check if child (j) is inside parent (i)
                iof = self.calculate_iof(boxes[j], boxes[i])

                if iof > self.iof_threshold:
                    # MERGE into 'i' (Union)
                    boxes[i][0] = min(boxes[i][0], boxes[j][0])
                    boxes[i][1] = min(boxes[i][1], boxes[j][1])
                    boxes[i][2] = max(boxes[i][2], boxes[j][2])
                    boxes[i][3] = max(boxes[i][3], boxes[j][3])
                    
                    keep[j] = False  # Remove absorbed box
                    
                    self.processing_log.append({
                        'rule': 'merge_union',
                        'action': 'merge',
                        'parent': i,
                        'child': j,
                        'iof': iof
                    })

        # Filter results
        filtered_boxes = [boxes[i] for i in range(n_boxes) if keep[i]]
        filtered_scores = [scores[i] for i in range(n_boxes) if keep[i]]
        filtered_class_ids = [class_ids[i] for i in range(n_boxes) if keep[i]]
        
        return filtered_boxes, filtered_scores, filtered_class_ids
