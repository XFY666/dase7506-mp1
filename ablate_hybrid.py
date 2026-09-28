"""Validation ablations of one frozen set of neural weights and mixture scalars.

Removed mechanisms are disabled without retuning the remaining settings. This
separates mechanism evidence from the wider validation search. Only validation
is scored; fixed benchmark files are hashed without using test predictions.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from common import PROTOCOL, ROOT, make_model, setup, sha
from evaluate import score
from evidence import benchmark_hashes
from tune_cache import validation_data


def ablation_variants(selected):
    """Disable each active mechanism independently, retaining all other scalars."""
    no_counts = dict(ngram_weight=0., ngram_early_weight=None,
                     ngram_early_tokens=0, ngram_confidence_power=0.)
    variants = [
        ('neural_raw', dict(no_counts, cache_weight=0., logit_temperature=1.)),
        ('neural_calibrated', dict(no_counts, cache_weight=0.)),
        ('neural_cache', no_counts),
        ('neural_counts', dict(cache_weight=0.)),
        ('full', {}),
    ]
    if selected.get('cache_min_similarity', -1.) > -1.:
        variants.append(('full_without_similarity_gate', dict(cache_min_similarity=-1.)))
    if selected.get('cache_layer', -1) != -1:
        variants.append(('full_with_final_layer_cache', dict(cache_layer=-1)))
    early = selected.get('ngram_early_weight')
    early_active = (int(selected.get('ngram_early_tokens', 0)) > 0
                    and early is not None
                    and float(early) != float(selected.get('ngram_weight', 0.)))
    if early_active:
        variants.append(('full_without_early_count_weight',
                         dict(ngram_early_weight=None, ngram_early_tokens=0)))
    count_active = float(selected.get('ngram_weight', 0.)) > 0. or (early_active and float(early) > 0.)
    if count_active and float(selected.get('ngram_confidence_power', 0.)) > 0.:
        variants.append(('full_without_count_confidence', dict(ngram_confidence_power=0.)))
    return variants


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file to preserve evidence.')
    source_names = ['student.py', 'hybrid.py', 'ngram_expert.py', 'evaluate.py',
                    'common.py', 'model.py', 'tune_cache.py', 'ablate_hybrid.py', 'evidence.py']
    sources = {name: sha(ROOT/name) for name in source_names}
    benchmark = benchmark_hashes()
    checkpoint_sha = sha(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] != 'hybrid':
        raise ValueError('Use a selected hybrid checkpoint from this protocol.')
    device, _ = setup(args.device, 'fp32', args.threads)
    selected = dict(checkpoint['config'])
    model, _ = make_model('hybrid', selected, device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    tokens, byte_count = validation_data()
    variants = ablation_variants(selected)
    rows = []
    started = time.perf_counter()
    for name, changes in variants:
        model.config = dict(selected, **changes)
        result = score(model, tokens, byte_count, device, 'fp32')
        result.pop('window_nll_nats')
        row = dict(name=name, changes=changes, **result)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if model.ngram_expert is not None:
        result = score(model.ngram_expert, tokens, byte_count, device, 'fp32')
        result.pop('window_nll_nats')
        rows.append(dict(name='counts_only', changes=None, **result))
    if (sources != {name: sha(ROOT/name) for name in source_names}
            or sha(args.checkpoint) != checkpoint_sha or benchmark_hashes() != benchmark):
        raise RuntimeError('Source or checkpoint changed during the ablation.')
    result = dict(protocol=PROTOCOL, split='validation', test_used=False,
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                  selected_config=selected, source_sha256=sources,
                  benchmark_sha256=benchmark,
                  precision='fp32', device=str(device), threads=args.threads,
                  setting_policy='Disable each named mechanism; do not retune any remaining scalar.',
                  new_training_targets=0, seconds=time.perf_counter()-started, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
