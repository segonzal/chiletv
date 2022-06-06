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
from video_reader import BatchedVideoReader, VideoReader
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


def find_batch_size(width: int, height: int, detector: FaceDetector, max_batch_size: int = np.inf):
    # increase batch size x2 until error
    batch_size = 1
    while True:
        batch_size = min(2 * batch_size, max_batch_size)
        try:
            image = (255 * np.random.random((batch_size, width, height, 3))).astype(np.uint8)
            detector(image)
        except (RuntimeError, MemoryError) as err:
            break
        else:
            # if max_batch_size was supported previously then it's the maximum possible valid value
            if batch_size >= max_batch_size:
                return max_batch_size

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


def detect_faces_on_video(reader, detector):
    # Get the time spent detecting and tracking boxes
    reader.start()
    width, height = reader.get_shape()
    data = {
        'frame_rate': reader.frame_rate,
        'batch_size': reader.batch_size,
        'min_face_size': detector.min_face_size,
        'max_frame_size': detector.max_frame_size,
        'frame_scale': detector.scale,
        'width': width,
        'height': height,
        'video_length': reader.get_duration(),
        'time': [],
        'content_delta': [],
        'bounding_box': [],
        'key_points': []
    }
    try:
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
    except RuntimeError as err:
        reader.clear_queue()
        raise err
    finally:
        reader.stop()
    return data


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


@argh.arg('video_path', help='Source folder for the detections.')
@argh.arg('--min-face-size', type=int, default=20, help='Minimum size of a face required by the face detector.')
@argh.arg('--max-frame-size', type=int, default=None, help='Max size for a frame.')
@argh.arg('--frame-scale', type=float, default=1.0, help='Scaling factor for all frames.')
@argh.arg('--use-cpu', action='store_true', help='Whether the face detector should use the CPU.')
def get_max_batch_size(video_path: str,
                       min_face_size: int = 20,
                       max_frame_size: int = None,
                       frame_scale: float = 1.0,
                       use_cpu: bool = False):
    detector = FaceDetector(min_face_size, max_frame_size, not use_cpu, frame_scale)

    stream = cv2.VideoCapture(video_path)
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    batch_size = find_batch_size(width, height, detector)
    print(f'{video_path} [{width}x{height}]: {batch_size}')
    ok, frame = stream.read()
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stream.release()


@argh.arg('src_folder', help='Source file or folder with the videos.')
@argh.arg('dst_folder', help='Destination folder for the detections.')
@argh.arg('--frame-rate', type=float, default=30.0, help='Frame rate to read videos.')
@argh.arg('--batch-size', type=int, default=0, help='Batch size for the face detector.')
@argh.arg('--min-face-size', type=int, default=20, help='Minimum size of a face required by the face detector.')
@argh.arg('--max-frame-size', type=int, default=None, help='Max size for a frame.')
@argh.arg('--frame-scale', type=float, default=1.0, help='Scaling factor for all frames.')
@argh.arg('--use-cpu', action='store_true', help='Whether the face detector should use the CPU.')
@argh.arg('-r', '--randomize', action='store_true', help='Randomize the order of files.')
@argh.arg('--max-batch-size', type=int, default=1024, help='Maximum batch size.')
@argh.arg('--max-retries', type=int, default=5, help='Maximum number of retries per video.')
def detect_faces(src_folder: str,
                 dst_folder: str,
                 frame_rate: float = 30.0,
                 batch_size: Union[int, str] = 'auto',
                 min_face_size: int = 20,
                 max_frame_size: int = None,
                 frame_scale: float = 1.0,
                 use_cpu: bool = False,
                 randomize: bool = False,
                 max_batch_size: int = 1024,
                 max_retries: int = 5):
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

    detector = FaceDetector(min_face_size, max_frame_size, not use_cpu, frame_scale)

    with tqdm.tqdm(ongoing_videos, total=len(all_videos), initial=len(done_videos)) as main_loop:
        for video_path in main_loop:
            main_loop.set_description(video_path.name)

            video_scale = frame_scale
            video_batch_size = batch_size

            reader = BatchedVideoReader(frame_rate)

            try:
                reader.open(video_path)
                width, height = reader.get_shape()

                if detector.max_frame_size and detector.max_frame_size < max(width, height):
                    video_scale = float(detector.max_frame_size) / float(frame_scale * max(width, height))

                detector.set_scale(video_scale)

                if video_batch_size <= 0:
                    video_batch_size = find_batch_size(width, height, detector, max_batch_size=max_batch_size)

                bz_frac = max(int(0.1 * video_batch_size), 1)
                for retry_num in range(max(1, max_retries)):
                    reader.set_batch_size(video_batch_size - bz_frac * retry_num)
                    try:
                        data = detect_faces_on_video(reader, detector)
                    except RuntimeError as err:
                        message = 'Retry {}: GPU Memory error for video "{}" with batch size {}'
                        main_loop.write(message.format(retry_num+1, video_path, reader.batch_size))
                    else:
                        # Write detection file
                        with (dst_folder / f'{video_path.stem}.detections.json').open('w', encoding='utf8') as wp:
                            json.dump(data, wp, cls=NumpyEncoder)
                        break
            except (cv2.error, ZeroDivisionError) as err:
                main_loop.write(f'Video "{video_path}"({reader.batch_size}) has errors.\n\n{str(err)}\n\n')
                continue

            del reader


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
    argh.dispatch_commands([sample_videos, get_max_batch_size, detect_faces, track_detections])
