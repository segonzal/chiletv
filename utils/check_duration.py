import subprocess
from pathlib import Path
from typing import List, Tuple

import argh
import pandas as pd


def get_duration(file: Path) -> float:
    cmd = f'ffprobe -v quiet -of default=nw=1:nk=1 -show_entries format=duration {file}'
    p = subprocess.run(cmd, capture_output=True)
    duration = p.stdout.strip()
    duration = float(duration) if duration not in (b'', b'N/A') else 0.0
    return duration


def get_shape(file: Path) -> Tuple[int, int]:
    cmd = f'ffprobe -v quiet -of default=nw=1:nk=1 -show_entries stream=width,height {file}'
    p = subprocess.run(cmd, capture_output=True)
    r = p.stdout.strip().split()
    if len(r) == 0:
        return 0, 0
    width, height = r
    width = int(width)
    height = int(height)
    return width, height


def read_data(videos: List[str], path: Path):
    for video_id in videos:
        data = {'video_id': video_id}
        for key, ext in [('original', '.mp4'), ('video', '.mp4'), ('audio', '.wav')]:
            data[key] = get_duration(path / key / (video_id + ext))
        data['width'], data['height'] = get_shape(path / f'original/{video_id}.mp4')
        yield data


@argh.arg('data-folder', type=str, help='Folder with the data.')
@argh.arg('download-folder', type=str, help='Folder with the downloaded originals.')
@argh.arg('csv-file', type=str, help='CSV file to save results.')
def main(data_folder: str, download_folder: str, csv_file: str):
    data_folder = Path(data_folder)
    download_folder = Path(download_folder)
    csv_file = Path(csv_file)

    processed_videos = []
    df = []
    if csv_file.exists():
        old_df = pd.read_csv(csv_file)
        processed_videos.extend(old_df['video_id'].values)
        df.append(old_df)

    video_ids = [v.stem for v in (data_folder / 'video').glob('**/*.mp4') if v.stem not in processed_videos]

    if len(video_ids):
        new_df = pd.DataFrame(read_data(video_ids, data_folder))
        df.append(new_df)

    df = pd.concat(df, sort=False)

    # Delete video and audio, move original and info to downloads
    bad_select = df.isin([0]).any(axis=1)
    for video_id in df[bad_select]['video_id']:
        print(video_id)

        # delete
        (data_folder / 'video' / f'{video_id}.mp4').unlink()
        (data_folder / 'audio' / f'{video_id}.wav').unlink()

        # move
        (download_folder / video_id).mkdir(exist_ok=True)
        (data_folder / 'info' / f'{video_id}.info.json').rename(download_folder / video_id / 'original.info.json')
        (data_folder / 'original' / f'{video_id}.mp4').rename(download_folder / video_id / 'original.mp4')

    df = df[~bad_select]

    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    argh.dispatch_command(main)