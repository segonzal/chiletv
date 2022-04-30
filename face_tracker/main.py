import json
import time
import random
from pathlib import Path

import argh
import tqdm

from utils import *
from face_detector import FaceDetector
from video_reader import BatchedVideoReader


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


@argh.arg('src_folder', help='Source folder for the videos.')
@argh.arg('dst_folder', help='Destination folder for the detections.')
@argh.arg('--frame-rate', default=30.0, help='Frame rate to read videos.')
@argh.arg('--batch-size', default=1024, help='Batch size for the face detector.')
@argh.arg('--min-face-size', default=20, help='Minimum size of a face required by the face detector.')
@argh.arg('--frame-size', default=640, help='Max size for a frame.')
@argh.arg('--use-cpu', action='store_true', help='Whether the face detector should use the CPU.')
def detect_faces(src_folder: str,
                 dst_folder: str,
                 frame_rate: float = 30.0,
                 batch_size: int = 1024,
                 min_face_size: int = 20,
                 frame_size: int = 640,
                 use_cpu: bool = False):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)

    dst_folder.mkdir(exist_ok=True)

    all_videos = list(src_folder.glob('**/*.mp4'))
    done_videos = set(video_id(v.name) for v in dst_folder.glob('**/*.detections.json'))
    ongoing_videos = [v for v in all_videos if video_id(v.name) not in done_videos]

    reader = BatchedVideoReader(frame_rate, batch_size)
    detector = FaceDetector(min_face_size, not use_cpu)

    with tqdm.tqdm(sorted(ongoing_videos), total=len(all_videos), initial=len(done_videos)) as main_loop:
        for video_path in main_loop:
            main_loop.set_description(video_path.name)

            data = dict(frame_rate=frame_rate,
                        batch_size=batch_size,
                        min_face_size=min_face_size,
                        frame_size=frame_size)
            try:
                reader.start(str(video_path))
                width, height = reader.get_shape()

                detector.set_scale(float(max(width, height)) / float(frame_size))

                data['width'] = width
                data['height'] = height
                data['video_length'] = reader.get_duration()
                data['time'] = []
                data['content_delta'] = []
                data['bounding_box'] = []
                data['key_points'] = []

                # Get the time spent detecting and tracking boxes
                start_time = time.time()
                with tqdm.tqdm(total=int(reader.get_duration()), leave=False) as mini_loop:
                    prev_descriptor = 0
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
                main_loop.write('An error has occurred for video', video_path)
                continue

            # Write detection file
            with (dst_folder / f'{video_path.stem}.detections.json').open('w', encoding='utf8') as wp:
                json.dump(data, wp, indent=4, cls=NumpyEncoder)


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

    all_detections = list(src_folder.glob('**/*.detections.json'))
    done_detections = set(video_id(v.name) for v in dst_folder.glob('**/*.tracks.json'))
    ongoing_detections = [v for v in all_detections if video_id(v.name) not in done_detections]

    with tqdm.tqdm(sorted(ongoing_detections), total=len(all_detections), initial=len(done_detections)) as main_loop:
        for detection_path in main_loop:
            main_loop.set_description(video_id(detection_path))


if __name__ == "__main__":
    argh.dispatch_commands([sample_videos, detect_faces, track_detections])
