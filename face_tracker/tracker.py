import numpy as np


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
                 max_gap_length: float,
                 min_shot_length: float):
        self.content_threshold = content_threshold
        self.iou_threshold = iou_threshold
        self.max_gap_length = max_gap_length
        self.min_shot_length = min_shot_length

        self.track_data = {}
        self.tracked_ids = []
        self.last_id = 0
        self.last_descriptor = 0

    def add_points_to_track(self, timestamp, bounding_box, key_points, track_id=None):
        if track_id is None:
            track_id = self.last_id
            self.last_id += 1
            self.track_data[track_id] = dict(time=[], bounding_box=[], key_points=[])

        track = self.track_data[track_id]
        track['time'].append(timestamp)
        track['bounding_box'].append(bounding_box)
        track['key_points'].append(key_points)

        # Updates the index to the last non-None element added.
        if bounding_box is not None:
            track['idx'] = len(track['bounding_box']) - 1

        return track_id

    def finish_track(self, track_id):
        track = self.track_data[track_id]

        # Removes the tail of undetected elements
        track['time'] = track['time'][:track['idx']]
        track['bounding_box'] = track['bounding_box'][:track['idx']]
        track['key_points'] = track['key_points'][:track['idx']]

        del track['idx']

        track['end_time'] = track['time'][0]
        track['start_time'] = track['time'][-1]
        track['duration'] = track['end_time'] - track['start_time']

        if track['duration'] < self.min_shot_length:
            del self.track_data[track_id]

    def finish_all_tracks(self):
        for i in self.tracked_ids:
            self.finish_track(i)
        self.tracked_ids.clear()

    def get_last_bounding_box(self, track_id: int):
        track = self.track_data[track_id]
        return track['bounding_box'][track['idx']]

    def get_last_updated_time(self, track_id):
        track = self.track_data[track_id]
        return track['time'][track['idx']]

    def close_track_by_time_up(self, timestamp):
        """Removes the tracks which last update is greater than the maximum allowed gap."""
        i = 0
        while i < len(self.tracked_ids):
            if timestamp - self.get_last_updated_time(self.tracked_ids[i]) > self.max_gap_length:
                self.finish_track(self.tracked_ids[i])
                del self.tracked_ids[i]
            else:
                i += 1

    def match_tracks(self, timestamp, new_boxes, new_points):
        """Matches boxes between frames by IOU."""
        tracked_idx = set(range(len(self.tracked_ids)))
        new_box_idx = set(range(len(new_boxes)))

        # Get the IOU for all pairs
        iou_mat = np.zeros((len(self.tracked_ids), len(new_boxes)), dtype=float)
        for i in tracked_idx:
            for j in new_box_idx:
                iou_mat[i, j] = iou(self.get_last_bounding_box(self.tracked_ids[i]), new_boxes[j])

        # Pick the detection with greatest iou
        while iou_mat.size != 0:
            i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)

            # Once the iou is bellow threshold then no valid detections are left
            if iou_mat[i, j] < self.iou_threshold:
                break

            # Assign box i to track j
            self.add_points_to_track(timestamp, new_boxes[j], new_points[j], track_id=self.tracked_ids[i])

            # Remove the pair from the candidates
            iou_mat[i, :] = -1
            iou_mat[:, j] = -1
            tracked_idx.remove(i)
            new_box_idx.remove(j)

        # Add an empty spot for tracks with gap
        for i in tracked_idx:
            self.add_points_to_track(timestamp, None, None, track_id=self.tracked_ids[i])

        # Return indices of newly added tracks
        for j in new_box_idx:
            self.add_points_to_track(timestamp, new_boxes[j], new_points[j], None)

    def detect_shot_transition(self, content_delta: float):
        if content_delta < self.content_threshold:
            self.finish_all_tracks()

    def update(self, timestamp, content_delta, new_boxes, new_points):
        self.close_track_by_time_up(timestamp)
        self.detect_shot_transition(content_delta)
        self.match_tracks(timestamp, new_boxes, new_points)

    def reset(self):
        self.track_data.clear()
        self.last_descriptor = 0
        self.tracked_ids.clear()
        self.last_id = 0

    def get_data(self):
        return {
            'content_threshold': self.content_threshold,
            'iou_threshold': self.iou_threshold,
            'max_gap_length': self.max_gap_length,
            'min_shot_length': self.min_shot_length,
            'tracks': self.track_data,
        }
