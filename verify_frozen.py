"""Check every inference dependency against a saved predictor freeze."""
import argparse
import json
from pathlib import Path
import zipfile

from evidence import PROTOCOL, ROOT, checked_relative_path, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'checkpoints/final/freeze.json')
    args = parser.parse_args()
    record = json.loads(args.manifest.read_text(encoding='utf-8'))
    if record.get('protocol') != PROTOCOL:
        raise ValueError('Unexpected freeze protocol.')
    expected = dict(record['inference_source_sha256'])
    expected.update(record['benchmark_sha256'])
    expected.update(record['freeze_tool_sha256'])
    expected.update({item['path']: item['sha256'] for item in record['assets']})
    expected.update({item['path']: item['sha256'] for item in record['evidence']})
    for name, digest in expected.items():
        path = checked_relative_path(name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'Frozen inference dependency missing or changed: {name}')
    counted = 0
    for item in record['assets']:
        path = checked_relative_path(item['path'])
        stored = path.stat().st_size
        expanded = stored
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                expanded = sum(member.file_size for member in archive.infolist())
        size = max(stored, expanded)
        if size != item['counted_uncompressed_bytes'] or stored != item['file_bytes']:
            raise ValueError(f'Frozen asset size mismatch: {item["path"]}')
        counted += size
    if counted != record['inference_assets_uncompressed_bytes'] or counted > 64*1024**2:
        raise ValueError('Frozen asset accounting is inconsistent or exceeds 64 MiB.')
    print(json.dumps(dict(freeze=str(args.manifest), verified_files=len(expected),
                          frozen_at_utc=record['frozen_at_utc'],
                          inference_assets_uncompressed_bytes=record['inference_assets_uncompressed_bytes'],
                          status='all hashes match'), indent=2))


if __name__ == '__main__':
    main()
