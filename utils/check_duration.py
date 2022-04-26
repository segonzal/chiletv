import csv
import argh
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Tuple


def get_duration(file: Path) ->Tuple[int, int, float]:
    show_entries = 'format=duration'

    if file.ext == '.mp4':
        show_entries += ':stream=width,height'
    
    p = subprocess.run(['ffprobe', '-v', 'quiet', '-of', 'default=nw=1:nk=1',
        '-show_entries', show_entries, str(video)], capture_output=True)

    ret = p.stdout.strip().split()

    width = int(ret[0]) if len(ret)==3 else 0
    height = int(ret[1]) if len(ret)==3 else 0
    duration = float(ret[-1]) if ret[-1] not in (b'', b'N/A') else 0.0
    
    return width, height, duration


@argh.atg('src', type=str, help='Folder with the videos.')
@argh.atg('dst', type=str, help='Folder where to store the original files once converted.')
@argh.atg('--prefix', type=str, default='original-', help='Prefix to rename the original files.')
@argh.atg('--pattern', type=str, default='**/*.mp4', help='Pattern of the files to search.')
def main(src_folder: str, csv_file: str, prefix: str='original-', pattern: str='**/*.mp4'):
    src_folder = Path(src_folder)
    csv_file = Path(csv_file)
    
    write_header = not csv_file.exists()

    with csv_file.open('a', newline='') as csvfile:
        fieldnames = ['video_id', 'width', 'height', 'duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', dialect='excel', extrasaction='ignore')
        
        if writeheader:
            writer.writeheader()
        
        for file in csv_file.glob(pattern):
            row = {}
            row['video_id'] = file.stem[len(prefix):] if file.stem.startswith(prefix) else file.stem
            row['width'], row['height'], row['duration'] = get_duration(file)
            writer.writerow(row)


if __name__ == '__main__':
    argh.dispatchc_ommand(main)
