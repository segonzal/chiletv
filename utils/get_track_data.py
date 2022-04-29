import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_point_data(track_folder: Path):
    data_time = []
    data_bbox = []
    data_tlen = []

    for track_file in Path(track_folder).glob('**/*.tracks.json'):
        with track_file.open('r', encoding='utf8') as fp:
            track_file = json.load(fp)

        # Ignore videos without tracks
        if len(track_file['tracks']) == 0:
            continue

        track_time, track_bbox = zip(*[(t['time'], t['bbox']) for t in track_file['tracks'].values()])

        track_time = list(track_time)
        track_bbox = list(track_bbox)
        track_tlen = []

        for i, (t, b) in enumerate(zip(track_time, track_bbox)):        
            # All tracks start from 0
            t = np.float32(t) - t[0]
            track_time[i] = t.tolist()

            # Length of a track in seconds
            track_tlen.append(t[-1])

            # Box positions from screen center
            b = np.float32(b).reshape(-1, 2, 2)
            b[:, :, 0] -= 0.5 * track_file['width']
            b[:, :, 1] -= 0.5 * track_file['height']
            track_bbox[i] = b.tolist()

        data_time.extend(track_time)
        data_bbox.extend(track_bbox)
        data_tlen.extend(track_tlen)
    return data_time, data_bbox, data_tlen


@argh.arg('save_path', help='Path to store the results.')
@argh.arg('track_folder', help='Folder with the track records.')
def main(save_path: str, track_folder: str):
    save_path =  Path(save_path)
    track_folder = Path(track_folder)

    data_time, data_bbox, data_tlen = load_point_data(track_folder)
    
    with save_path.open('wb') as fp:
        pickle.dump((data_time, data_bbox, data_tlen), fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    argh.dispatch_command(main)
