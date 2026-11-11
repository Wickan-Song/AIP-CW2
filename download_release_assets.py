"""Download the public Release parts, reconstruct the ZIPs, and restore the project."""

import argparse
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

from restore_assets import ASSETS, ROOT, digest, restore

BASE = 'https://github.com/Wickan-Song/AIP-CW2/releases/download/v0.1.0/'
PART_SIZE = 32 * 1024 * 1024
ARCHIVES = {'plantseg.zip': 1057281724, 'AIP-CW2-weights.zip': 1167712096}


def get_part(cache, name, size, offline):
    destination = cache / name
    if destination.is_file() and destination.stat().st_size == size:
        return destination
    if offline:
        raise FileNotFoundError(f'Missing or incomplete local part: {destination}')
    partial = destination.with_suffix(destination.suffix + '.partial')
    print(f'Downloading {name}', flush=True)
    request = Request(BASE + name, headers={'User-Agent': 'AIP-CW2-restore/1.0'})
    with urlopen(request, timeout=120) as response, partial.open('wb') as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if partial.stat().st_size != size:
        raise ValueError(f'Incorrect length for {name}. Run the command again to retry it.')
    partial.replace(destination)
    return destination


def reconstruct(cache, archive, size, offline):
    target = cache / archive
    if target.is_file() and digest(target) == ASSETS[archive]:
        return target
    partial = target.with_suffix('.zip.partial')
    count = (size + PART_SIZE - 1) // PART_SIZE
    paths = [get_part(cache, f'{archive}.part{index + 1:02}', min(PART_SIZE, size - PART_SIZE*index), offline)
             for index in range(count)]
    with partial.open('wb') as out:
        for path in paths:
            with path.open('rb') as part:
                shutil.copyfileobj(part, out, length=1024 * 1024)
    if digest(partial) != ASSETS[archive]:
        raise ValueError(f'Reconstructed {archive} has a different SHA-256. Check the downloaded parts.')
    partial.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, default=ROOT/'.release_downloads')
    parser.add_argument('--offline', action='store_true', help='Use the parts already present in --parts-dir.')
    parser.add_argument('--archives-only', action='store_true', help='Reconstruct ZIPs without extracting them.')
    options = parser.parse_args()
    cache = options.parts_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    expected = {}
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        checksum, name = line.split('  ', 1)
        expected[name] = checksum
    for archive, size in ARCHIVES.items():
        source = reconstruct(cache, archive, size, options.offline)
        print(f'Verified {source}', flush=True)
        if options.archives_only:
            continue
        is_data = archive == 'plantseg.zip'
        if is_data:
            files = {name: checksum for name, checksum in expected.items() if name.startswith('data/plantseg/')}
        else:
            files = {name: checksum for name, checksum in expected.items()
                     if name.startswith('artifacts/') and name.endswith('.pt')}
        restore(source, archive, files, is_data)


if __name__ == '__main__':
    main()
