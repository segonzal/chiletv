import argh
import json
import random
from tqdm import tqdm
from typing import List
from pathlib import Path
from itertools import chain

def read_urls(name):
	urls = set()
	with open(name, 'r') as  fp:
		for line in fp:
			urls.add(line.strip())
	return urls


def write_urls(file: Path, urls, shuffle=False, sort=False):
	urls = list(urls)

	if shuffle:
		random.shuffle(urls)
	elif sort:
		urls.sort()

	with file.open('w', encoding='utf-8') as fp:
		for url in urls:
			fp.write(url + '\n')

def get_url(folder):
	vinfo_file = folder / (folder.name + '.info.json')
	video_file = folder / (folder.name + '.mp4')

	# print(folder, vinfo_file, video_file)

	if video_file.exists() and vinfo_file.exists():
		try:
			with vinfo_file.open('r', encoding='utf-8') as fp:
				return json.load(fp)['webpage_url']
		except (UnicodeDecodeError, json.decoder.JSONDecodeError) as e:
			pass
	return None


@argh.arg('sources', type=str, nargs='+', help="Video storage paths")
@argh.arg('-i', '--source-file', type=str, help="URL soruce file")
def move(sources: List[str], source_file: str = 'urls-raw-all.txt'):
	""" Move all downloaded elements at the end of the file.
	"""
	source_file = Path(source_file).absolute()
	urls_sources = set(read_urls(source_file))

	folders = list(chain(*(Path(s).glob('*') for s in sources)))
	loop = tqdm(folders)
	urls_downloaded = dict()
	for folder in loop:
		loop.set_description(str(folder.name))
		url = get_url(folder)
		if url is None:
			root = folder.parent.parent
			trash = root / 'trash'
			trash.mkdir(parents=True, exist_ok=True)
			folder.rename(trash / folder.name)
		else:
			urls_downloaded[url] = urls_downloaded.get(url, []) + [folder]
	print(f"There are {len(urls_downloaded):d}/{len(folders):d}[{float(len(urls_downloaded))/len(folders): 3.0%}] correctly downloaded videos.")

	for url, folders in urls_downloaded.items():
		if len(folders) != 1:
			print(f"Video {url} is duplicated: {', '.join(folders)}")

	urls_downloaded = set(urls_downloaded.keys())

	urls_sources = sorted(urls_sources)
	i = 0
	total = len(urls_sources)
	while i < total:
		if urls_sources[i] in urls_downloaded:
			urls_sources.append(urls_sources.pop(i))
			total -= 1
		else:
			i += 1


	write_urls(source_file, urls_sources, shuffle=False, sort=False)


@argh.arg('archive-file', type=str, help="Archive file")
@argh.arg('sources', type=str, nargs='+', help="Video storage paths")
def check(archive_file: str, sources: List[str]):
	archive_file = Path(archive_file).absolute()
	#urls_sources = read_urls('urls-all.txt')
	
	#urls_discard = read_urls('urls-filtered.txt').union(read_urls('urls-unavailable.txt'))

	folders = list(chain(*(Path(s).glob('*') for s in sources)))
	loop = tqdm(folders)

	urls_downloaded = dict()
	for folder in loop:
		loop.set_description(str(folder))

		url = get_url(folder)
		#loop.set_postfix_str(str(url))

		if url is None:
			root = folder.parent.parent
			trash = root / 'trash'
			trash.mkdir(parents=True, exist_ok=True)
			folder.rename(trash / folder.name)
		else:
			urls_downloaded[url] = urls_downloaded.get(url, []) + [folder]
			
	print(f"There are {len(urls_downloaded):d}/{len(folders):d}[{float(len(urls_downloaded))/len(folders): 3.0%}] correctly downloaded videos.")

	for url, folders in urls_downloaded.items():
		if len(folders) != 1:
			print(f"Video {url} is duplicated: {', '.join(folders)}")

	urls_downloaded = set(urls_downloaded.keys())

	urls_archive = {url.replace('https://www.youtube.com/watch?v=', 'youtube ') for url in urls_downloaded}
	urls_archive = urls_archive.union(read_urls(archive_file))
	urls_archive.discard('')
	write_urls(archive_file, urls_archive, sort=True)

	#urls_archive = urls_downloaded.union(urls_discard)
	#urls_sources.difference_update(urls_archive)
	
	#write_urls('urls-downloaded.txt', urls_downloaded, shuffle=False)
	#write_urls('urls-sources.txt', urls_sources, shuffle=False)


@argh.arg('sources', type=str, nargs='+', help="Video storage paths")
def rename(sources: List[str]):
	video = list(chain(*(Path(s).glob('**/*.tracks.json') for s in sources)))
	loop = tqdm(video)
	for video in loop:
		folder = video.parent
		folder.rename(folder.parent.parent / 'detected' / (video.stem[:-len('.tracks')]))
		# try:
		# except OSError as e:
		#  	# Should delete contents first
		# 	folder.rmdir()


if __name__ == '__main__':
	argh.dispatch_commands([check,move,rename])
