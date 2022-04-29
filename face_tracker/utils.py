import cv2
import numpy as np


def get_content_descriptor(frame, shape=(8,8)):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), shape, interpolation=cv2.INTER_AREA).flatten()


def get_content_descriptor_distance(descriptor_a, descriptor_b):
    return np.sqrt(np.sum((descriptor_b - descriptor_a) ** 2))
