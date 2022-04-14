import argh
import subprocess
from tqdm import tqdm
from pathlib import Path
from multiprocessing.pool import ThreadPool


def rename_originals(src: str, num: int = 0, prefix:str='original-'):
	src = Path(src)

	files = sorted(src.glob('**/' + prefix + '*.mp4'))

	if num > 0:
		files = files[:num]

	for file in files:
		out = file.parent / file.stem[len(prefix):] / file.name
		print(out)
		file.rename(out)


def work(input_file: Path, output_name: str, dst: Path):
	cmd = [
		'ffmpeg', '-i', str(input_file), '-v', 'quiet',
		'-map', '0:a', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', str(input_file.parent / (output_name + '.wav')),
		'-map', '0:v', '-vcodec', 'h264', '-filter:v', 'fps=30', str(input_file.parent / (output_name + '.mp4'))
	]
	print(input_file)
	#loop.set_description(input_file.name())
	#loop.update(1)
	p = subprocess.Popen(cmd)
	p.wait()
	input_file.rename(dst / (output_name + '.mp4'))


@argh.arg('src', help='Source folder.')
@argh.arg('dst', help='Destination for processed original files.')
@argh.arg('num', type=int, help='Number of processes.')
@argh.arg('--prefix', type=str, default='original-', help='Prefix to rename original files.')
@argh.arg('--rename', type=int, default=None, const=0, nargs='?', help='The number of original files to rename. Empty to all.')
def main(src: str, dst:str, num: int, prefix: str = 'original-', rename: int=None):

	# if rename is not None:
	# 	rename_originals(src, rename, prefix)

	src = Path(src)
	dst = Path(dst)
	all_files = list(src.glob('**/' + prefix + '*.mp4'))
	tp = ThreadPool(num)
	# loop = tqdm(total=len(all_files))
	for file in all_files:
		output_name = file.stem[len(prefix):]
		tp.apply_async(work, (file, output_name, dst))
	try:
		tp.close()
		tp.join()
	except KeyboardInterrupt:
		tp.terminate()


if __name__ == '__main__':
	argh.dispatch_command(main)
