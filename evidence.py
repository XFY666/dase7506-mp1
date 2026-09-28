"""Small torch-free helpers for reproducible evaluation evidence."""
import hashlib
import json
from pathlib import Path


PROTOCOL = '7506-mp1-wt2-v2'
ROOT = Path(__file__).resolve().parent
INFERENCE_SOURCES = ('student.py', 'hybrid.py', 'ngram_expert.py',
                     'common.py', 'evaluate.py', 'model.py')
BENCHMARK_FILES = ('data/manifest.json', 'data/tokenizer.json',
                   'data/wikitext_train.txt', 'data/wikitext_validation.txt',
                   'data/wikitext_test.txt')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(names, root=ROOT):
    return {name: sha(root/name) for name in names}


def benchmark_hashes(root=ROOT):
    """Hash fixed bytes, including test; never tokenize or evaluate test here."""
    hashes = file_hashes(BENCHMARK_FILES, root)
    manifest = json.loads((root/'data/manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('Unexpected benchmark protocol.')
    for name, expected in manifest['sha256'].items():
        if hashes.get('data/'+name) != expected:
            raise ValueError(f'Changed benchmark file: {name}')
    return hashes


def snapshot(checkpoint, root=ROOT):
    return dict(checkpoint_sha256=sha(checkpoint),
                source_sha256=file_hashes(INFERENCE_SOURCES, root),
                benchmark_sha256=benchmark_hashes(root))


def checked_relative_path(name, root=ROOT):
    relative = Path(name)
    path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f'Dependency path escapes the submission: {name}')
    return path
