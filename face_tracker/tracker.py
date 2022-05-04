import numpy as np
from utils import get_content_descriptor, get_content_descriptor_distance


def iou(bbox_a: np.float32, bbox_b: np.float32) -> float:
    right = max(bbox_a[0], bbox_b[0])
    top = max(bbox_a[1], bbox_b[1])
    left = min(bbox_a[2], bbox_b[2])
    bottom = min(bbox_a[3], bbox_b[3])

    inter_area = max(0.0, left - right) * max(0.0, bottom - top)

    bbox_a_area = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    bbox_b_area = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    union_area = bbox_a_area + bbox_b_area - inter_area

    return inter_area / float(union_area)


class Tracker:
    def __init__(self,
                 content_threshold: float,
                 iou_threshold: float,
                 max_gap_length: float):
        self.content_threshold = content_threshold
        self.iou_threshold = iou_threshold
        self.max_gap_length = max_gap_length

        self.tracked_boxes = []
        self.last_id = 0
        self.last_descriptor = 0

    def filter_by_timestamp(self, timestamp):
        """Filters the boxes by timestamp in place."""
        i = 0
        while i < len(self.tracked_boxes):
            if timestamp - self.tracked_boxes[i]['end_timestamp'] > self.max_gap_length:
                del self.tracked_boxes[i]
            else:
                i += 1

    def get_bbox(self, index: int):
        return self.tracked_boxes[index]['bbox']

    def add_bbox(self, bbox: np.array, timestamp: float):
        self.tracked_boxes.append({
            'bbox': bbox,
            'start_timestamp': timestamp,
            'end_timestamp': timestamp,
            'id': self.last_id
        })
        self.last_id += 1
        return self.last_id - 1

    def update_bbox(self, index: int, bbox: np.array, timestamp: float):
        self.tracked_boxes[index]['bbox'] = bbox
        self.tracked_boxes[index]['end_timestamp'] = timestamp
        return self.tracked_boxes[index]['id']

    def match_boxes(self, new_boxes, timestamp):
        """Matches boxes between frames by IOU."""
        tracked_idx = set(range(len(self.tracked_boxes)))
        new_box_idx = set(range(len(new_boxes)))

        new_boxes_id = [-1] * len(new_boxes)

        iou_mat = np.zeros((len(self.tracked_boxes), len(new_boxes)), dtype=float)
        for i in tracked_idx:
            for j in new_box_idx:
                iou_mat[i, j] = iou(self.get_bbox(i), new_boxes[j])

        # Pick the detection with greatest iou
        while iou_mat.size != 0:
            i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)

            if iou_mat[i, j] < self.iou_threshold:
                break

            iou_mat[i, :] = -1
            iou_mat[:, j] = -1

            new_boxes_id[j] = self.update_bbox(i, new_boxes[j], timestamp)

            tracked_idx.remove(i)
            new_box_idx.remove(j)

        # Add to register all new detections
        for j in new_box_idx:
            new_boxes_id[j] = self.add_bbox(new_boxes[j], timestamp)

        return new_boxes_id

    def detect_shot_transition(self, content_delta: float):
        if content_delta < self.content_threshold:
            self.tracked_boxes.clear()

    def reset(self):
        self.last_descriptor = 0
        self.tracked_boxes.clear()
        self.last_id = 0