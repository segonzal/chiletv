import json
from typing import List
from pathlib import Path

import argh
import pandas as pd
from tqdm import tqdm


def load_length_data(track_folders: List[Path]):
    for path in track_folders:

        with (path / 'config.json').open('r', encoding='utf8') as fp:
            config_file = json.load(fp)

        for track_file in path.glob('**/*.tracks.json'):
            with track_file.open('r', encoding='utf8') as fp:
                track_file = json.load(fp)
            
            for track in track_file['tracks'].values():
                yield (
                    track_file['url'][track_file['url'].rindex('=')+1:],
                    track_file['video_duration'],
                    track_file['width'],
                    track_file['height'],
                    track_file['detection_duration'],
                    config_file['frame_rate'],
                    config_file['content_threshold'],
                    config_file['iou_threshold'],
                    config_file['max_time_gap_length'],
                    config_file['min_shot_length'],
                    config_file['min_face_size'],
                    config_file['detector_scale'],
                    track['end_time'] - track['start_time'])


@argh.arg('save_path', help='Path to store the results.')
@argh.arg('track_folders', help='Folder with the track records.')
def main(save_path: str, track_folders: List[str]):
    save_path = Path(save_path)
    track_folders = [Path(f) for f in track_folders]

    video_data_columns = [
        'video_id',
        'video_length',
        'video_width',
        'video_height',
        'detection_length',
        'run_frame_rate',
        'run_content_threshold',
        'run_iou_threshold',
        'run_max_gap_length',
        'run_min_shot_length',
        'run_min_face_size',
        'run_detector_scale',
        'track_length']

    video_data = load_length_data(track_folders)
    
    df = pd.DataFrame.from_records(video_data, columns=video_data_columns)
    
    df.to_csv(save_path)


if __name__ == '__main__':
    argh.dispatch_command(main)
