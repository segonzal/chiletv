import subprocess
from itertools import islice
from pathlib import Path
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock

import argh
from tqdm import tqdm


def work(src_file: Path, video_id: str, dst_folder: Path, raw_folder: Path, lock: Lock, loop: tqdm):
	# Leve a register of wich file is being processed
	lock.acquire()
	loop.write(str(src_file))
	lock.release()

	cmd = [
		'ffmpeg', '-i', str(src_file), '-v', 'quiet',
		'-map', '0:a', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', str(src_file.parent / (video_id + '.wav')),
		'-map', '0:v', '-vcodec', 'h264', '-filter:v', 'fps=30', str(src_file.parent / (video_id + '.mp4'))
	]

	p = subprocess.Popen(cmd)
	p.wait()

	# Move the original video to raw
	src_file.rename(raw_folder / (video_id + '.mp4'))
	
	# Move the parent folder to dst
	src_file.parent.rename(dst_folder / video_id)

	# Update progress bar
	lock.acquire()
	loop.update()
	lock.release()


@argh.arg('src_folder', help='Folder with the videos to recode.')
@argh.arg('dst_folder', help='Destination for recoded videos.')
@argh.arg('raw_folder', help='Destination for original videos.')
@argh.arg('-n', '--num', type=int, default=0, help='Number of videos.')
@argh.arg('-p', '--proc', type=int, default=1, help='Number of processes.')
@argh.arg('-x', '--prefix', type=str, default='original-', help='Prefix to rename original files.')
def main(src_folder: str, dst_folder: str, raw_folder: str, num: int=0, proc: int=1, prefix: str='original-'):
	# src: str, dst:str, num: int, prefix: str = 'original-', rename: int=None):
	src_folder = Path(src_folder)
	dst_folder = Path(dst_folder)
	raw_folder = Path(raw_folder)

	source_files = src_folder.glob('**/*.mp4')

	if num != 0:
		source_files = islice(source_files, num)

	source_files = list(source_files)

	tp = ThreadPool(proc)
	tl = Lock()
	loop = tqdm(total=len(source_files))

	try:
		for src_file in source_files:
			video_id = src_file.stem

			# Renames the original video if it was not renamed already
			if not src_file.name.startswith(prefix):
				new_name = src_file.parent / (prefix + src_file.name)
				if new_name.exists():
					raise Exception('File "{new_name}" already exists')
				src_file.rename(new_name)
				src_file = new_name
			else:
				video_id = video_id[len(prefix):]

			tp.apply_async(work, (src_file, video_id, dst_folder, raw_folder, tl, loop))
		
		tp.close()
		tp.join()
	except KeyboardInterrupt:
		tp.terminate()
	finally:
		print('Done!')


if __name__ == '__main__':
	argh.dispatch_command(main)
