import argh
import shutil
import multiprocessing.dummy
import subprocess
from pathlib import Path
from typing import List, Set
from tempfile import TemporaryDirectory


# youtube-dl --config-location youtube-dl.conf
#            --batch-file .\batch.txt
#            --download-archive "archive.txt"
#            --output "%(id)s.%(ext)s"
#            --no-progress


def merge_archives(files: List[Path]) -> Set[str]:
    """Reads the contents of all the given files and returns a set with their contents."""
    archived_lines = set()
    for archive_file in files:
        with archive_file.open('r', encoding='utf8') as fp:
            for line in fp:
                archived_lines.add(line.strip())

    archived_lines.discard('')
    return archived_lines


def write_sets(data: Set[str], num: int, folder: Path, fmt: str) -> List[Path]:
    """Writes the set as a text file."""
    files_path = [(folder / fmt.format(i)) for i in range(num)]

    files = [f.open('w', encoding='utf8') for f in files_path]

    for i, string in enumerate(data):
        files[i % num].write(string + '\n')

    [f.close() for f in files]

    return files_path


def duplicate_file(file: Path, num: int, folder: Path, fmt: str) -> List[Path]:
    """Makes an exact copy of a file n times."""
    files_path = [(folder / fmt.format(i)) for i in range(num)]
    for out_file in files_path:
        shutil.copy2(file, out_file)
    return files_path


def download(v):
    arr = [
        'youtube-dl',
        '--config-location',
        v['config_location'],
        '--batch-file',
        v['batch_file'], '--download-archive',
        v['archive'], '--no-progress',
        '--output',
        v['output_folder'] + "/%(id)s/%(id)s.%(ext)s"
    ]
    print(*arr)
    subprocess.check_call(arr)


@argh.arg('config-location', type=str, help="youtube-dl config file")
@argh.arg('batch-file', type=str, help="batch file with urls")
@argh.arg('archive-file', type=str, help="archive file with urls")
@argh.arg('output-folder', type=str, help="Video storage folder")
@argh.arg('number', type=int, help="Number of parallel processes")
def main(config_location: str, batch_file: str, archive_file: str, output_folder: str, number: int):
    output_folder = Path(output_folder)
    batch_file = Path(batch_file)
    archive_file = Path(archive_file)

    # Merge previous files
    batch_urls = merge_archives([batch_file])
    archive_urls = merge_archives([archive_file])
    archive_urls = {url.replace('youtube ', 'https://www.youtube.com/watch?v=') for url in archive_urls}

    tmp_dir = TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)

    # Make separate files for each thread
    list_batch_files = write_sets(batch_urls.difference(archive_urls), number, tmp_path, 'batch-{0:02d}.txt')
    list_archive_files = duplicate_file(archive_file, number, tmp_path, 'archive-{0:02d}.txt')

    # Run all the parallel processes of youtube-dl
    arr = [{'batch_file': str(bf),
            'archive': str(af),
            'output_folder': str(output_folder),
            'config_location': config_location}
           for bf, af in zip(list_batch_files, list_archive_files)]

    try:
        p = multiprocessing.dummy.Pool(number)
        p.map(download, arr)
    finally:
        archive_urls = merge_archives(list_archive_files)

        with archive_file.open('w', encoding='utf8') as fp:
            for line in archive_urls:
                fp.write(line + '\n')

        tmp_dir.cleanup()


if __name__ == "__main__":
    argh.dispatch_commands([main])
