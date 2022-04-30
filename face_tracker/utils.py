import cv2
import json
import numpy as np


def get_content_descriptor(frame, shape=(8,8)):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), shape, interpolation=cv2.INTER_AREA).flatten()


def get_content_descriptor_distance(descriptor_a, descriptor_b):
    return np.sqrt(np.sum((descriptor_b - descriptor_a) ** 2))


def remove_empty_detections(data, keep_ids):
    """Removes the frames with no detections from the registry."""
    i = 0
    while i < len(data):
        timestamp, bounding_box, key_points, face_ids = data[i].values()

        j = 0
        while j < len(face_ids):
            if face_ids[j] in keep_ids:
                j += 1
            else:
                del bounding_box[j]
                del key_points[j]
                del face_ids[j]

        if len(bounding_box) == 0:
            del data[i]
        else:
            i += 1


def video_id(filename: str):
    return filename[:filename.index('.')]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
