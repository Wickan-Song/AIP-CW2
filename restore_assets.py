"""Restore the original data and frozen weights from the GitHub release assets."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
ASSETS = {
    'plantseg.zip': 'cbdfb273eb26fea93fda52bbfe4766cc31881feb3a1202b13e4d85e8a4c0bcfa',
    'AIP-CW2-weights.zip': '64a46da8f9a3e2b2b8216afd932f97437a35746862c0ef648607ff488518073b',
}


def digest(path):
    with path.open('rb') as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
        return value.hexdigest()


def restore(archive_path, asset_name, expected, data):
    if digest(archive_path) != ASSETS[asset_name]:
        raise ValueError(f'{asset_name} does not match the release checksum.')
    with ZipFile(archive_path) as archive:
        prefix = 'data/' if data else ''
        members = {prefix + info.filename: info for info in archive.infolist() if not info.is_dir()}
        missing = expected.keys() - members.keys()
        if missing:
            raise ValueError(f'{asset_name} is missing {len(missing)} expected files.')
        for name, checksum in expected.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Invalid relative path: {name}')
            path = ROOT.joinpath(*relative.parts)
            if path.exists():
                if digest(path) != checksum:
                    raise ValueError(f'Existing file differs from this release: {name}')
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temp = Path(handle.name)
                with archive.open(members[name]) as member:
                    shutil.copyfileobj(member, handle)
            try:
                if digest(temp) != checksum:
                    raise ValueError(f'Checksum mismatch: {name}')
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
    print(f'{asset_name}: restored or verified {len(expected):,} files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plantseg_zip', type=Path)
    parser.add_argument('weights_zip', type=Path)
    args = parser.parse_args()
    expected = {}
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        checksum, name = line.split('  ', 1)
        expected[name] = checksum
    data = {name: checksum for name, checksum in expected.items() if name.startswith('data/plantseg/')}
    weights = {name: checksum for name, checksum in expected.items() if name.startswith('artifacts/') and name.endswith('.pt')}
    restore(args.plantseg_zip, 'plantseg.zip', data, True)
    restore(args.weights_zip, 'AIP-CW2-weights.zip', weights, False)
