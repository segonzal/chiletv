from typing import List, Tuple

import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self, min_face_size: int, use_gpu: bool, scale: float = 1.0):
        device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(min_face_size=min_face_size, keep_all=True, device=device)
        self.scale = scale

    def __call__(self, frame_batch: List[np.array]) -> Tuple[List[np.array], List[np.array]]:
        frames = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, fx=self.scale, fy=self.scale)
                  for frame in frame_batch if frame is not None]
        bbox_batch, _, kpts_batch = self.model.detect(frames, landmarks=True)

        bbox_batch = [b / self.scale if b is not None else [] for b in bbox_batch]
        kpts_batch = [p / self.scale if p is not None else [] for p in kpts_batch]

        return bbox_batch, kpts_batch
