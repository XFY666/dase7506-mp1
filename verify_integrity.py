"""Verify the frozen benchmark and original baseline against release hashes."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'PACKAGE_MANIFEST.json').read_text())
    fixed = ['common.py', 'evaluate.py', 'model.py', 'train.py',
             'configs/baseline.json', 'requirements.txt', 'tests/test_contract.py']
    fixed += [key.removeprefix('code/') for key in manifest if key.startswith('code/data/')]
    results = []
    for relative in fixed:
        expected = manifest['code/'+relative]
        actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Original benchmark/baseline file changed: {relative}')
        results.append({'path': relative, 'sha256': actual})
    print(json.dumps({'all_unchanged': True, 'files': results}, indent=2))


if __name__ == '__main__':
    main()
