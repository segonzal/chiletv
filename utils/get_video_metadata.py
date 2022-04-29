import subprocess
from typing import List
from pathlib import Path
from itertools import chain

import argh
import pandas as pd
from tqdm import tqdm


def get_video_data(video: Path, prefix='original-'):
    name = video.stem
    p = subprocess.run(['ffprobe', '-v', 'quiet', '-of', 'default=nw=1:nk=1',
                        '-show_entries', 'format=duration:stream=width,height',
                        str(video)], capture_output=True)
    width, height, duration = p.stdout.strip().split()
    name = name[len(prefix):] if name.startswith(prefix) else name
    width = int(width)
    height = int(height)
    duration = float(duration) if duration not in (b'', b'N/A') else 0.0
    return name, width, height, duration

@argh.arg('video_csv', help='CSV file.')
@argh.arg('video_src', nargs='+', help='Source folders for the videos.')
@argh.arg('-x', '--prefix', type=str, default='original-', help='Prefix to rename original files.')
def main(video_csv: str, video_src: List[str], prefix: str = 'original-'):
    video_csv = Path(video_csv)

    video_files = [Path(p).glob('**/*.mp4') for p in video_csv]
    loop = tqdm(chain(*video_files))
    
    data = (get_video_data(video, prefix) for video in loop)
    column = ['video_id', 'width', 'height', 'duration', ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(video_csv)


if __name__ == '__main__':
    argh.dispatch_command(main)
