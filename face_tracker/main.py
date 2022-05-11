import json
import time
import random
from pathlib import Path
from typing import Union

import argh
import tqdm
import numpy as np

from utils import *
from face_detector import FaceDetector
from video_reader import BatchedVideoReader
from tracker import Tracker


def get_detections(reader: BatchedVideoReader, detector: FaceDetector):
    """Detects faces and its key points for each batch of frames."""
    for frame_batch, timestamp_batch in reader.read_batch():
        bounding_box_batch, key_points_batch = detector(frame_batch)
        for frame, timestamp, bounding_box, key_points in zip(frame_batch,
                                                              timestamp_batch,
                                                              bounding_box_batch,
                                                              key_points_batch):
            yield frame, timestamp, bounding_box, key_points


@argh.arg('src_folder', help='Source folder for the detections.')
@argh.arg('dst_folder', help='Destination folder for the tracks.')
@argh.arg('sample_size', help='Sample size.')
@argh.arg('--seed', help='Seed for the RNG.')
def sample_videos(src_folder: str,
                  dst_folder: str,
                  sample_size: int,
                  seed: int = 0):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)

    dst_folder.mkdir(exist_ok=True)
    all_files = list(src_folder.glob('**/*.mp4'))

    random.seed(seed)
    loop = tqdm.tqdm(random.sample(all_files, sample_size))

    for file_path in loop:
        loop.set_description(file_path.name)
        file_path.rename(dst_folder / file_path.name)


def find_batch_size(width: int, height: int, detector: FaceDetector):
    # increase batch size x2 until error
    batch_size = 1
    while True:
        try:
            image = (255 * np.random.random((batch_size, width, height, 3))).astype(np.uint8)
            detector(image)
        except RuntimeError as err:
            break
        batch_size *= 2

    upper_bound = batch_size
    lower_bound = batch_size // 2
    # The upper bound is error, the lower correct
    # get the middle value, if error: try again
    while upper_bound - lower_bound > 2:
        batch_size = (upper_bound + lower_bound) // 2
        try:
            image = (256 * np.random.random((batch_size, width, height, 3))).astype(np.uint8)
            detector(image)
        except RuntimeError as err:
            upper_bound = batch_size
        else:
            lower_bound = batch_size
    return lower_bound


@argh.arg('src_folder', help='Source file or folder with the videos.')
@argh.arg('dst_folder', help='Destination folder for the detections.')
@argh.arg('--frame-rate', default=30.0, help='Frame rate to read videos.')
@argh.arg('--batch-size', default='auto', help='Batch size for the face detector.')
@argh.arg('--min-face-size', default=20, help='Minimum size of a face required by the face detector.')
@argh.arg('--max-frame-size', default=None, help='Max size for a frame.')
@argh.arg('--use-cpu', action='store_true', help='Whether the face detector should use the CPU.')
@argh.arg('-r', '--randomize', action='store_true', help='Randomize the order of files.')
def detect_faces(src_folder: str,
                 dst_folder: str,
                 frame_rate: float = 30.0,
                 batch_size: Union[int, str] = 'auto',
                 min_face_size: int = 20,
                 max_frame_size: int = None,
                 use_cpu: bool = False,
                 randomize: bool = False):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)

    dst_folder.mkdir(exist_ok=True)

    if src_folder.is_file():
        done_videos = []
        ongoing_videos = all_videos = [src_folder]
    else:
        all_videos = list(src_folder.glob('**/*.mp4'))
        done_videos = set(video_id(v.name) for v in dst_folder.glob('**/*.detections.json'))
        ongoing_videos = sorted([v for v in all_videos if video_id(v.name) not in done_videos])
        
    if randomize:
        random.shuffle(ongoing_videos)

    reader = BatchedVideoReader(frame_rate)
    detector = FaceDetector(min_face_size, not use_cpu)
    detector.set_scale(1.0)

    if batch_size.isnumeric():
        batch_size = int(batch_size)

    with tqdm.tqdm(ongoing_videos, total=len(all_videos), initial=len(done_videos)) as main_loop:
        for video_path in main_loop:
            main_loop.set_description(video_path.name)
            curr_batch_size = batch_size
            curr_scale = 1.0

            try:
                reader.open(str(video_path))
                width, height = reader.get_shape()

                if max_frame_size and max_frame_size < max(width, height):
                    curr_scale = float(max_frame_size) / float(max(width, height))

                detector.set_scale(curr_scale)
                
                if curr_batch_size == 'auto':
                    curr_batch_size = min(1024, find_batch_size(width, height, detector))

                reader.set_batch_size(curr_batch_size)
                reader.start()

                data = {
                    'frame_rate': frame_rate,
                    'batch_size': reader.batch_size,
                    'min_face_size': min_face_size,
                    'max_frame_size': max_frame_size,
                    'width': width,
                    'height': height,
                    'video_length': reader.get_duration(),
                    'time': [],
                    'content_delta': [],
                    'bounding_box': [],
                    'key_points': []
                }

                # Get the time spent detecting and tracking boxes
                with tqdm.tqdm(total=int(reader.get_duration()), leave=False) as mini_loop:
                    mini_loop.set_postfix(batch_size=reader.batch_size)
                    prev_descriptor = 0
                    start_time = time.time()
                    for frame, timestamp, bounding_box, key_points in get_detections(reader, detector):
                        mini_loop.update(int(timestamp - mini_loop.n))

                        descriptor = get_content_descriptor(frame)
                        content_delta = get_content_descriptor_distance(descriptor, prev_descriptor)
                        prev_descriptor = descriptor

                        data['time'].append(timestamp)
                        data['content_delta'].append(content_delta)
                        data['bounding_box'].append(bounding_box)
                        data['key_points'].append(key_points)
                    end_time = time.time()
                    data['detection_length'] = end_time - start_time

            except (cv2.error, ZeroDivisionError) as err:
                main_loop.write(f'An error has occurred for video "{video_path}"')
                continue
            except RuntimeError as err:
                main_loop.write(f'GPU Memory error for video "{video_path}" with batch size: {reader.batch_size}')
                reader.stop()
                continue

            # Write detection file
            with (dst_folder / f'{video_path.stem}.detections.json').open('w', encoding='utf8') as wp:
                json.dump(data, wp, cls=NumpyEncoder)


@argh.arg('src_folder', help='Source folder for the detections.')
@argh.arg('dst_folder', help='Destination folder for the tracks.')
@argh.arg('--content-threshold', help='Threshold for the shot-transition detector.')
@argh.arg('--iou-threshold', help='Threshold for the IOU overlap between different-frame detections.')
@argh.arg('--max-gap-length', help='Maximum allowed gap in seconds between corresponding detections.')
@argh.arg('--min-shot-length', help='Minimum duration in seconds for a valid track.')
def track_detections(src_folder: str,
                     dst_folder: str,
                     content_threshold: float = 90.0,
                     iou_threshold: float = 0.5,
                     max_gap_length: float = 1.0,
                     min_shot_length: float = 10.0):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)

    dst_folder.mkdir(exist_ok=True)

    all_detections = list(src_folder.glob('**/*.detections.json'))
    done_detections = set(video_id(v.name) for v in dst_folder.glob('**/*.tracks.json'))
    ongoing_detections = [v for v in all_detections if video_id(v.name) not in done_detections]

    tracker = Tracker(content_threshold, iou_threshold, max_gap_length, min_shot_length)

    with tqdm.tqdm(sorted(ongoing_detections), total=len(all_detections), initial=len(done_detections)) as main_loop:
        for detection_path in main_loop:
            main_loop.set_description(video_id(detection_path.name))

            with detection_path.open('r', encoding='utf8') as fp:
                detection_data = json.load(fp)

            detection_data = zip(detection_data['time'],
                                 detection_data['content_delta'],
                                 detection_data['bounding_box'],
                                 detection_data['key_points'])

            for timestamp, content_delta, bounding_box, key_points in detection_data:
                tracker.update(timestamp, content_delta, bounding_box, key_points)
            tracker.finish_all_tracks()

            # Write detection file
            with (dst_folder / f'{video_id(detection_path.name)}.tracks.json').open('w', encoding='utf8') as wp:
                json.dump(tracker.get_data(), wp, cls=NumpyEncoder)

            tracker.reset()


if __name__ == "__main__":
    argh.dispatch_commands([sample_videos, detect_faces, track_detections])
