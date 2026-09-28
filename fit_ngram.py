"""Fit the compact modified Kneser-Ney expert using supplied training text only.

Discounts follow Chen and Goodman (1999), Section 3, Equation 17:
https://u.cs.biu.ac.il/~yogo/courses/mt2014/papers/chen-goodman-99.pdf
No external implementation or language-model weights are loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer

VOCAB = 2048
ROOT = Path(__file__).resolve().parent
PROTOCOL = '7506-mp1-wt2-v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded_grams(tokens, order):
    keys = np.zeros(len(tokens)-order+1, dtype=np.int64)
    for shift in range(order):
        keys = keys*VOCAB + tokens[shift:len(tokens)-order+shift+1]
    return keys


def fit(tokens, max_order=5, min_count=2):
    if not 2 <= max_order <= 5 or min_count < 1:
        raise ValueError('Use orders 2 through 5 and a positive minimum count.')
    raw = {order: np.unique(encoded_grams(tokens, order), return_counts=True)
           for order in range(2, max_order+1)}
    unigram = np.bincount(raw[2][0] % VOCAB, minlength=VOCAB).astype(np.float32) + .1
    unigram /= unigram.sum()
    arrays = {'unigram': unigram}
    statistics = []
    for order in range(2, max_order+1):
        keys, counts = raw[order]
        if order < max_order:
            keys, counts = np.unique(raw[order+1][0] % (VOCAB ** order), return_counts=True)
        contexts, first = np.unique(keys // VOCAB, return_index=True)
        totals = np.add.reduceat(counts, first).astype(np.float32)
        frequencies = [int(np.sum(counts == c)) for c in range(1, 5)]
        n1, n2, n3, n4 = frequencies
        if min(n1, n2, n3) == 0:
            raise ValueError('Training corpus lacks count frequencies needed for modified discounts.')
        y = n1 / (n1 + 2*n2)
        discounts = [1-2*y*n2/n1, 2-3*y*n3/n2, 3-4*y*n4/n3]
        if any(not 0 < value < limit for value, limit in zip(discounts, [1, 2, 3])):
            raise ValueError('Estimated discounts are outside their valid ranges.')
        keep = counts >= (1 if order == 2 else min_count)
        keys, counts = keys[keep], counts[keep].astype(np.float32)
        locations = np.searchsorted(contexts, keys // VOCAB)
        discount = np.where(counts == 1, discounts[0], np.where(counts == 2, discounts[1], discounts[2]))
        direct = (np.maximum(counts-discount, 0.) / totals[locations]).astype(np.float32)
        mass = np.bincount(locations, weights=direct, minlength=len(contexts)).astype(np.float32)
        used_context = mass > 0
        used_entry = direct > 0
        arrays.update({f'n{order}_keys': keys[used_entry],
                       f'n{order}_direct': direct[used_entry],
                       f'n{order}_contexts': contexts[used_context],
                       f'n{order}_backoff': 1.-mass[used_context]})
        statistics.append(dict(order=order, entries=int(used_entry.sum()), contexts=int(used_context.sum()),
                               count_frequencies_1_to_4=frequencies, modified_discounts=discounts))
    return arrays, statistics


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-order', type=int, default=5)
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--max-asset-mib', type=float, default=24.)
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must have the .npz extension.')
    metadata_path = args.output.with_suffix('.fit.json')
    if args.output.exists() or metadata_path.exists():
        parser.error('Choose a new output path to preserve existing assets and provenance.')
    manifest = json.loads((args.data_dir/'manifest.json').read_text(encoding='utf-8'))
    inputs = {}
    for name in ['tokenizer.json', 'wikitext_train.txt']:
        digest = sha(args.data_dir/name)
        if digest != manifest['sha256'][name]:
            raise ValueError(f'Changed benchmark file: {name}')
        inputs[name] = digest
    tokenizer = Tokenizer.from_file(str(args.data_dir/'tokenizer.json'))
    tokens = np.asarray(tokenizer.encode((args.data_dir/'wikitext_train.txt').read_bytes().decode('utf-8')).ids,
                        dtype=np.int64)
    prepared = time.perf_counter()
    arrays, statistics = fit(tokens, args.max_order, args.min_count)
    fit_seconds = time.perf_counter()-prepared
    array_bytes = sum(value.nbytes for value in arrays.values())
    if array_bytes > args.max_asset_mib * 1024**2:
        raise ValueError('Count arrays exceed the requested asset budget.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    metadata = dict(protocol=PROTOCOL, algorithm='pruned modified Kneser-Ney', source_split='train',
                    validation_used=False, test_used=False, external_training_data=False,
                    train_tokens=len(tokens), available_next_token_targets=len(tokens)-1,
                    counts_derived_from_one_training_sequence=True,
                    max_order=args.max_order, min_count=args.min_count, unigram_additive_floor=.1,
                    preparation_seconds=prepared-started, fit_seconds=fit_seconds,
                    process_seconds=time.perf_counter()-started, statistics=statistics,
                    array_bytes=array_bytes, asset_bytes=args.output.stat().st_size,
                    asset_sha256=sha(args.output), input_sha256=inputs, builder_sha256=sha(Path(__file__)),
                    reference='https://u.cs.biu.ac.il/~yogo/courses/mt2014/papers/chen-goodman-99.pdf')
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
