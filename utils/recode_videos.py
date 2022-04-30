import subprocess
from itertools import islice
from pathlib import Path
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock

import argh
from tqdm import tqdm


def work(video_folder: Path, dataset_folder: Path, lock: Lock, loop: tqdm):
    video_id = video_folder.name

    # Leave a record of which file is being processed
    lock.acquire()
    loop.write(video_id)
    lock.release()

    paths = {
        'info': video_folder / 'original.info.json',
        'original': video_folder / 'original.mp4',
        'video': video_folder / 'video.mp4',
        'audio': video_folder / 'audio.wav',
    }

    cmd = 'ffmpeg -i {original} -v quiet' + \
          '-map 0:a -acodec pcm_s16le -ac 1 -ar 16000 {audio}' + \
          '-map 0:v -vcodec h264 -filter:v fps=30 {video}'
    cmd = cmd.format_map(paths)

    p = subprocess.Popen(cmd)
    p.wait()

    # Move completed files
    for key, path in paths.items():
        path.rename(dataset_folder / key / (video_id + path.name[path.name.index('.'):]))

    # Delete emptied video folder
    try:
        video_folder.rmdir()
    except OSError as err:
        lock.acquire()
        loop.write(f'Error: {video_folder} is not empty.')
        lock.release()

    # Update progress bar
    lock.acquire()
    loop.update()
    lock.release()


@argh.arg('downloads_folder', help='Folder with the videos to recode.')
@argh.arg('dataset_folder', help='Destination for recoded videos.')
@argh.arg('-n', '--num', type=int, default=0, help='Number of videos.')
@argh.arg('-p', '--proc', type=int, default=1, help='Number of processes.')
def main(downloads_folder: str, dataset_folder: str, num: int = 0, proc: int = 1):
    downloads_folder = Path(downloads_folder)
    dataset_folder = Path(dataset_folder)

    video_folders = downloads_folder.glob('*')

    if num != 0:
        video_folders = islice(video_folders, num)

    video_folders = list(video_folders)

    tp = ThreadPool(proc)
    tl = Lock()
    loop = tqdm(total=len(video_folders))

    try:
        for video_folder in video_folders:
            tp.apply_async(work, (video_folder, dataset_folder, tl, loop))
        tp.close()
        tp.join()
    except KeyboardInterrupt:
        tp.terminate()
    finally:
        loop.write('Done!')


if __name__ == '__main__':
    argh.dispatch_command(main)
