import numpy as np
from typing import List

from utils import iou


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

        self.tracks = {}
        self.opened_tracks = []
        self.next_id = 0
        self.last_shot_timestamp = 0

    def reset(self):
        self.tracks.clear()
        self.opened_tracks.clear()
        self.next_id = 0
        self.last_shot_timestamp = 0

    def add_new_track(self) -> int:
        track_id = self.next_id
        self.next_id += 1

        self.tracks[track_id] = dict(time=[], bounding_box=[], key_points=[])
        self.opened_tracks.append(track_id)
        return track_id

    def update_track(self, track_id: int, timestamp: float, bounding_box: List, key_points: List):
        self.tracks[track_id]['time'].append(timestamp)
        self.tracks[track_id]['bounding_box'].append(bounding_box)
        self.tracks[track_id]['key_points'].append(key_points)

        # Updates the index to the last non-None element added.
        if bounding_box is not None:
            self.tracks[track_id]['length'] = len(self.tracks[track_id]['bounding_box'])

    def get_track_bounding_box(self, track_id: int) -> List:
        item_num = self.tracks[track_id]['length']
        return self.tracks[track_id]['bounding_box'][item_num - 1]

    def get_track_timestamp(self, track_id: int) -> float:
        item_num = self.tracks[track_id]['length']
        return self.tracks[track_id]['time'][item_num - 1]

    def finish_track(self, track_id: int):
        item_num = self.tracks[track_id]['length']
        end_time = self.tracks[track_id]['time'][0]
        start_time = self.tracks[track_id]['time'][item_num - 1]

        self.tracks[track_id]['time'] = self.tracks[track_id]['time'][:item_num]
        self.tracks[track_id]['bounding_box'] = self.tracks[track_id]['bounding_box'][:item_num]
        self.tracks[track_id]['key_points'] = self.tracks[track_id]['key_points'][:item_num]
        self.tracks[track_id]['end_time'] = end_time
        self.tracks[track_id]['start_time'] = start_time
        self.tracks[track_id]['duration'] = start_time - end_time

        self.opened_tracks.remove(track_id)

    def finish_all_tracks(self):
        for track_id in self.opened_tracks:
            self.finish_track(track_id)

    def get_data(self):
        data = {
            'content_threshold': self.content_threshold,
            'iou_threshold': self.iou_threshold,
            'max_gap_length': self.max_gap_length,
            'min_shot_length': self.min_shot_length,
            'tracks': self.tracks.copy()
        }
        return data

    def close_tracks_by_gap(self, timestamp: float):
        i = 0
        while i < len(self.opened_tracks):
            if timestamp - self.get_track_timestamp(self.opened_tracks[i]) > self.max_gap_length:
                self.finish_track(self.opened_tracks[i])
            else:
                i += 1

    def close_by_shot_transition(self, timestamp: float, content_delta: float):
        if content_delta < self.content_threshold and timestamp - self.last_shot_timestamp > self.min_shot_length:
            self.finish_all_tracks()
            self.last_shot_timestamp = timestamp

    def match_tracks(self, timestamp: float, bounding_box_list: List, key_points_list: List):
        len_opened_tracks = len(self.opened_tracks)
        len_bounding_boxes = len(bounding_box_list)

        opened_track_indices = set(range(len_opened_tracks))
        new_bounding_boxes_indices = set(range(len_bounding_boxes))

        # Get the IOU for all pairs
        iou_mat = np.zeros((len_opened_tracks, len_bounding_boxes), dtype=float)
        for i in opened_track_indices:
            for j in new_bounding_boxes_indices:
                track_bounding_box = self.get_track_bounding_box(self.opened_tracks[i])
                iou_mat[i, j] = iou(track_bounding_box, bounding_box_list[j])

        # Pick the detection with greatest iou
        while iou_mat.size != 0:
            i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)

            # Once the iou is bellow threshold then no valid detections are left
            if iou_mat[i, j] < self.iou_threshold:
                break

            # Assign box i to track j
            self.update_track(self.opened_tracks[i], timestamp, bounding_box_list[j], key_points_list[j])

            # Remove the pair from the candidates
            iou_mat[i, :] = -1
            iou_mat[:, j] = -1
            opened_track_indices.remove(i)
            new_bounding_boxes_indices.remove(j)

        # Add None for tracks with gaps
        for i in opened_track_indices:
            self.update_track(self.opened_tracks[i], timestamp, None, None)

        # Add new tracks
        for j in new_bounding_boxes_indices:
            self.update_track(self.add_new_track(), timestamp, bounding_box_list[j], key_points_list[j])

    def update(self, timestamp: float, content_delta: float, bounding_box_list: List, key_points_list: List):
        self.close_by_shot_transition(timestamp, content_delta)
        self.match_tracks(timestamp, bounding_box_list, key_points_list)
        self.close_tracks_by_gap(timestamp)
