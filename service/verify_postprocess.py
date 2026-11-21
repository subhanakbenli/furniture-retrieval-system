import unittest
from src.postprocess import PostProcessor

class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PostProcessor(iof_threshold=0.5)

    def test_iou(self):
        # Two identical boxes should have IoU 1.0
        box1 = [0, 0, 10, 10]
        self.assertAlmostEqual(self.processor.calculate_iou(box1, box1), 1.0)
        
        # Non-overlapping boxes
        box2 = [20, 20, 30, 30]
        self.assertEqual(self.processor.calculate_iou(box1, box2), 0.0)

    def test_iof(self):
        # Box inside another
        inner = [2, 2, 8, 8]  # area 36
        outer = [0, 0, 10, 10] # area 100
        
        # IoF should be 1.0 (intersection is entire inner box)
        self.assertAlmostEqual(self.processor.calculate_iof(inner, outer), 1.0)

    def test_conflict_resolution(self):
        # Two boxes, same spot, different classes -> Should Delete Both
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        
        boxes = [box1, box2]
        scores = [0.9, 0.8]
        classes = [0, 1] # Different classes
        
        f_boxes, _, _ = self.processor.apply_rules(boxes, scores, classes)
        self.assertEqual(len(f_boxes), 0, "Should have deleted conflicting boxes")

    def test_merge_rule(self):
        # Box 2 is inside Box 1 -> Should Merge
        box1 = [0, 0, 100, 100]
        box2 = [10, 10, 50, 50]
        
        boxes = [box1, box2]
        scores = [0.9, 0.8]
        classes = [0, 0] # Same class
        
        f_boxes, _, _ = self.processor.apply_rules(boxes, scores, classes, iof_threshold=0.5)
        
        self.assertEqual(len(f_boxes), 1, "Should have merged into one box")
        # Check union size (should be size of box1 since box2 is inside)
        self.assertEqual(f_boxes[0], [0, 0, 100, 100])

if __name__ == '__main__':
    unittest.main()
