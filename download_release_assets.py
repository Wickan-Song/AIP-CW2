"""Download and verify the public CW2 Release assets, then restore the project."""

import subprocess
import sys
from urllib.request import Request, urlopen
import shutil

from restore_assets import ASSETS, ROOT, digest

BASE = 'https://github.com/Wickan-Song/AIP-CW2/releases/download/v0.1.0/'


def download(directory, name, expected):
    target = directory / name
    if target.is_file() and digest(target) == expected:
        return target
    partial = target.with_suffix('.zip.partial')
    print(f'Downloading {name}', flush=True)
    request = Request(BASE + name, headers={'User-Agent': 'AIP-CW2-restore/1.0'})
    with urlopen(request, timeout=120) as response, partial.open('wb') as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if digest(partial) != expected:
        raise ValueError(f'Checksum mismatch for {name}.')
    partial.replace(target)
    return target


if __name__ == '__main__':
    cache = ROOT / '.release_downloads'
    cache.mkdir(exist_ok=True)
    paths = [download(cache, name, checksum) for name, checksum in ASSETS.items()]
    subprocess.run([sys.executable, str(ROOT / 'restore_assets.py'), *map(str, paths)], check=True)
