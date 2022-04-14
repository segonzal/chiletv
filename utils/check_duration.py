import csv
import argh
import subprocess
from datetime import timedelta
from pathlib import Path


@argh.atg('src', type=str, help='Folder with the videos.')
@argh.atg('dst', type=str, help='Folder where to store the original files once converted.')
@argh.atg('--prefix', type=str, default='original-', help='Prefix to rename the original files.')
def main(src: str, dst: str, prefix:str='original-'):
    files = [
        ('original', dst, '.mp4'),
        ('video', src, '.mp4'),
        ('audio', src, '.wav')
    ]
    data = {}
    for key, path, fmt in files:
        for file in Path(path).glob('**/*' + fmt):
            if file.name.startswith(prefix):
                continue

            p = subprocess.run(['ffprobe', '-v', 'quiet', '-of', 'csv=p=0', '-show_entries', 'format=duration', str(file)], capture_output=True)
            
            duration = p.stdout.strip()
            duration = float(duration) if duration not in (b'', b'N/A') else 0.0
            
            name = file.stem[len(prefix):] if file.stem.startswith(prefix) else file.stem

            data[name] = data.get(name, {'name': name})
            data[name][key] = duration
            data[name]['path-' + key] = file

    src = Path(src)
    dst = Path(dst)

    with (dst / 'names.csv').open('w', newline='') as csvfile:
        fieldnames = ['name', 'original', 'video', 'audio', 'ok']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', dialect='excel', extrasaction='ignore')
        writer.writeheader()
        for value in data.values():
            value['ok'] = all([
                value.get('video', 0) != 0,
                value.get('audio', 0) != 0,
                abs(value.get('original', 0) - value.get('video', 0)) < 1,
                abs(value.get('original', 0) - value.get('audio', 0)) < 1
            ])
            if value['ok']:
                value['ok'] = ''
            else:
                print(value['name'])
                if value.get('path-original', False):
                    value['path-original'].rename(src / value['name'] / (prefix + value['name'] + '.mp4'))
                if value.get('path-video', False):
                    value['path-video'].unlink()
                if value.get('path-audio', False):
                    value['path-audio'].unlink()
            writer.writerow(value)


if __name__ == '__main__':
    argh.dispatchc_ommand(main)
