from typing import List, Tuple

import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self, min_face_size: int, max_frame_size: int, use_gpu: bool, scale: float = 1.0):
        self.min_face_size = min_face_size
        self.max_frame_size = max_frame_size
        self.use_gpu = use_gpu
        self.scale = scale

        device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(min_face_size=min_face_size, keep_all=True, device=device)

    def __call__(self, frame_batch: List[np.array]) -> Tuple[List[np.array], List[np.array]]:
        frames = [
            cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                None,
                fx=self.scale,
                fy=self.scale)
            for frame in frame_batch if frame is not None]

        bounding_box_batch, _, key_points_batch = self.model.detect(frames, landmarks=True)

        bounding_box_batch = [b / self.scale if b is not None else [] for b in bounding_box_batch]
        key_points_batch = [p / self.scale if p is not None else [] for p in key_points_batch]

        return bounding_box_batch, key_points_batch

    def set_scale(self, scale: float):
        self.scale = scale
