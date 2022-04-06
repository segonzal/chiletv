import cv2
import json
import time
import tqdm
import platform
from pathlib import Path

import argh
import torch
from video_reader import BatchedVideoReader
from face_detector import FaceDetector
from tracker import Tracker


def get_video_url(filename: Path) -> str:
    """Returns the url of the downloaded video."""
    filename = filename.parent / (filename.stem + '.info.json')
    with filename.open('r', encoding='utf8') as fp:
        obj = json.load(fp)
        return obj['webpage_url']


def get_detections(reader: BatchedVideoReader, detector: FaceDetector):
    """Detects faces and its keypoints for each batch of frames."""
    for frame_batch, timestamp_batch in reader.read_batch():
        bbox_batch, kpts_batch = detector(frame_batch)
        for frame, timestamp, bbox, kpts in zip(frame_batch, timestamp_batch, bbox_batch, kpts_batch):
            yield frame, timestamp, bbox, kpts


def get_data(tracker, reader, detector):
    """Gets the tracked data."""
    for frame, timestamp, bbox, kpts in get_detections(reader, detector):
        tracker.filter_by_timestamp(timestamp)
        tracker.detect_shot_transition(frame)
        curr_face_ids = tracker.match_boxes(bbox, timestamp)
        yield timestamp, bbox, kpts, curr_face_ids


def remove_empty_detections(data, keep_ids):
    """Removes the frames with no detections from the registry."""
    i = 0
    while i < len(data):
        timestamp, bbox, kpts, face_ids = data[i].values()

        j = 0
        while j < len(face_ids):
            if face_ids[j] in keep_ids:
                j += 1
            else:
                del bbox[j]
                del kpts[j]
                del face_ids[j]

        if len(bbox) == 0:
            del data[i]
        else:
            i += 1


@argh.arg('src', help='Source folder for the videos.')
@argh.arg('--dst', help='Destination folder for the tracks.')
@argh.arg('--frame-rate', help='Frame rate to read videos.')
@argh.arg('--batch-size', help='Batch size for the face detector.')
@argh.arg('--content-threshold', help='Threshold for the shot-transition detector.')
@argh.arg('--iou-threshold', help='Threshold for the IOU overlap between different-frame detections.')
@argh.arg('--max-time-gap-length', help='Maximum allowed gap in seconds between corresponding detections.')
@argh.arg('--min-shot-length', help='Minimum duration in seconds for a valid track.')
@argh.arg('--min-face-size', help='Minimum size of a face required by the face detector.')
@argh.arg('--detector-scale', help='Scaling factor for any image given to the face detector.')
@argh.arg('--use-gpu', help='Whether the face detector should use the GPU.')
def main(src: str,
         dst: str = None,
         frame_rate: float = 30.0,
         batch_size: int = 1024,
         content_threshold: float = 90.0,
         iou_threshold: float = 0.5,
         max_time_gap_length: float = 1.0,
         min_shot_length: float = 10.0,
         min_face_size: int = 20,
         detector_scale: float = 0.125,
         use_gpu: bool = True):
    root = Path(src)
    if dst is not None:
        dst = Path(dst)
        dst.mkdir(exist_ok=True)

    with (dst / 'config.json').open('w', encoding='utf8') as wp:
        json.dump(dict(frame_rate=frame_rate,
                       batch_size=batch_size,
                       content_threshold=content_threshold,
                       iou_threshold=iou_threshold,
                       max_time_gap_length=max_time_gap_length,
                       min_shot_length=min_shot_length,
                       min_face_size=min_face_size,
                       detector_scale=detector_scale,
                       use_gpu=use_gpu,
                       gpu_device_name=torch.cuda.get_device_name(0),
                       platform=platform.processor()), wp, indent=4)

    reader = BatchedVideoReader(frame_rate, batch_size)
    detector = FaceDetector(min_face_size, use_gpu, scale=detector_scale)
    tracker = Tracker(content_threshold, iou_threshold, max_time_gap_length)

    all_videos = list(root.glob('**/*.mp4'))
    pending_videos = [v for v in all_videos
                      if not ((v.parent if dst is None else dst) / (v.stem + '.tracks.json')).exists()]
    loop = tqdm.tqdm(sorted(pending_videos), total=len(all_videos), initial=len(all_videos) - len(pending_videos))

    for filename in loop:
        loop.set_description(str(filename))

        try:
            reader.start(str(filename))
            width, height = reader.get_shape()
            video_duration = reader.get_duration()
        except (cv2.error, ZeroDivisionError) as err:
            continue

        video_url = get_video_url(filename)
        tracker.reset()

        # Get the time spent detecting and tracking boxes
        start_time = time.time()
        data = list(get_data(tracker, reader, detector))
        end_time = time.time()

        # Remove all frames without detections or too short
        data = [dict(time=timestamp,
                     bbox=[b.flatten().tolist() for b in bbox],
                     kpts=[k.flatten().tolist() for k in kpts],
                     faceid=face_ids)
                for timestamp, bbox, kpts, face_ids in data]

        # Get metadata of each tracked face
        tracks = {}
        for frame_data in data:
            timestamp = frame_data['time']
            for face_id, bbox, kpts in zip(frame_data['faceid'], frame_data['bbox'], frame_data['kpts']):
                if face_id not in tracks:
                    tracks[face_id] = dict(start_time=timestamp, end_time=timestamp, data=[])
                else:
                    tracks[face_id]['end_time'] = max(timestamp, tracks[face_id]['end_time'])
                tracks[face_id]['data'].append(dict(time=timestamp, bbox=bbox, kpts=kpts))

        tracks = {fid: tms for fid, tms in tracks.items() if tms['end_time'] - tms['start_time'] > min_shot_length}

        remove_empty_detections(data, tracks)

        data = {
            'url': video_url,
            'frame_rate': frame_rate,
            'video_duration': video_duration,
            'detection_duration': end_time - start_time,
            'width': width,
            'height': height,
            'tracks': tracks,
        }

        json_dst = filename.parent if dst is None else dst
        with (json_dst / (filename.stem + '.tracks.json')).open('w', encoding='utf8') as wp:
            json.dump(data, wp, indent=1)


def get_sample(src, dst, num):
    import random
    random.seed(0)
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(exist_ok=True)
    all_files = list(src.glob('**/*.mp4'))
    loop = tqdm.tqdm(random.sample(all_files, num))
    for file in loop:
        file = file.parent
        loop.set_description(str(file))
        file.rename(dst / file.name)


if __name__ == "__main__":
    argh.dispatch_command(main)
