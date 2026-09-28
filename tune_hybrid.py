"""Select fixed neural/cache/count mixture scalars using validation only.

Per-target probabilities and hidden states exist only in memory for the current
batch. Saved evidence contains aggregate scores, settings, hashes, and ancestry.
The selected predictor is checked with the unchanged official validation scorer.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import torch

from common import PROTOCOL, ROOT, make_model, setup, sha, windows
from evaluate import score
from hybrid import count_mixing_coefficients
from tune_cache import number_list, validation_data, target_base_probabilities, target_cache_probabilities


def verify_sources_unchanged(expected):
    changed = [name for name, digest in expected.items() if sha(ROOT/name) != digest]
    if changed:
        raise RuntimeError('Source files changed during validation selection: '+', '.join(changed))


def accumulate_hybrid_nll(base, cache, active, counts, cache_weights, ngram_weights, valid, count_coefficients=None):
    """Aggregate NLL [temperature, cache setting, cache weight, count weight]."""
    result = torch.empty(base.shape[0], cache.shape[0], len(cache_weights), len(ngram_weights),
                         device=base.device, dtype=torch.float64)
    for cache_index, cache_weight in enumerate(cache_weights):
        coefficient = active.float() * cache_weight
        neural = base[:, None] * (1-coefficient[None]) + cache[None] * coefficient[None]
        for count_index, count_weight in enumerate(ngram_weights):
            mixing = count_weight if count_coefficients is None else count_coefficients[count_index][None, None]
            mixed = neural * (1-mixing) + counts[None, None] * mixing
            losses = -mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()
            result[:, :, cache_index, count_index] = losses.masked_fill(~valid[None, None], 0.).double().sum((-2, -1))
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ngram-asset', type=Path, default=Path('assets/ngram_mkn5.npz'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--logit-temperatures', default='.9,.95,1,1.05,1.1')
    parser.add_argument('--ngram-weights', default='0,.05,.1,.15,.2,.25,.3,.4,.5')
    parser.add_argument('--ngram-confidence-powers', default='0')
    parser.add_argument('--early-weights', default='same', help='Comma-separated weights or same for the ordinary weight.')
    parser.add_argument('--early-cutoffs', default='8,16,32')
    parser.add_argument('--cache-weights', default='0')
    parser.add_argument('--cache-temperatures', default='15')
    parser.add_argument('--cache-layer', type=int, default=-1,
                        help='-1 uses final normalized hidden states; 1..depth uses the raw post-block state.')
    parser.add_argument('--decays', default='0')
    parser.add_argument('--min-histories', default='1')
    parser.add_argument('--min-similarities', default='-1')
    args = parser.parse_args()
    temperatures = number_list(args.logit_temperatures)
    ngram_weights = number_list(args.ngram_weights)
    confidence_powers = number_list(args.ngram_confidence_powers)
    early_weights = [None if v == 'same' else float(v) for v in args.early_weights.split(',')]
    early_cutoffs = number_list(args.early_cutoffs, int)
    cache_weights = number_list(args.cache_weights)
    cache_temperatures = number_list(args.cache_temperatures)
    decays = number_list(args.decays)
    minimums = number_list(args.min_histories, int)
    similarities = number_list(args.min_similarities)
    if any(t <= 0 for t in temperatures) or any(t < 0 for t in cache_temperatures):
        parser.error('Logit temperatures must be positive; cache temperatures must be nonnegative.')
    if any(d < 0 for d in decays) or any(m < 1 for m in minimums):
        parser.error('Decay must be nonnegative; minimum history must be positive.')
    if any(not -1 <= value <= 1 for value in similarities):
        parser.error('Minimum cache similarities must be in [-1,1].')
    if any(not 0 <= w < 1 for w in ngram_weights+cache_weights+[w for w in early_weights if w is not None]) or args.batch_size < 1:
        parser.error('Weights must be in [0,1) and batch size must be positive.')
    if any(power < 0 for power in confidence_powers) or any(cutoff < 0 for cutoff in early_cutoffs):
        parser.error('Confidence powers and early cutoffs must be nonnegative.')
    count_settings = []
    for weight, power, early in itertools.product(ngram_weights, confidence_powers, early_weights):
        for cutoff in ([0] if early is None else early_cutoffs):
            count_settings.append(dict(ngram_weight=weight, ngram_confidence_power=power,
                                       ngram_early_weight=early, ngram_early_tokens=cutoff))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose a new empty output directory.')
    asset = (ROOT/args.ngram_asset).resolve()
    if not asset.is_relative_to(ROOT.resolve()):
        parser.error('The n-gram asset must be inside the submitted code directory.')
    relative_asset = asset.relative_to(ROOT.resolve()).as_posix()
    source_hashes = {name: sha(ROOT/name) for name in ['student.py', 'hybrid.py', 'ngram_expert.py',
                    'tune_hybrid.py', 'tune_cache.py', 'common.py', 'evaluate.py']}
    parent_sha = sha(args.checkpoint)
    asset_sha = sha(asset)
    device, _ = setup(args.device, 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] not in ('student', 'hybrid'):
        raise ValueError('Use a compatible student/hybrid checkpoint from this protocol.')
    config = dict(checkpoint['config'])
    if args.cache_layer != -1 and not 1 <= args.cache_layer <= config['depth']:
        parser.error('Cache layer must be -1 or an integer from 1 through model depth.')
    config.update(logit_temperature=1., cache_weight=0., cache_min_similarity=-1., ngram_weight=0.,
                  cache_layer=args.cache_layer,
                  ngram_confidence_power=0., ngram_early_weight=None, ngram_early_tokens=0,
                  ngram_asset=relative_asset, ngram_sha256=asset_sha)
    model, implementation_sha = make_model('hybrid', config, device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    tokens, byte_count = validation_data()
    use_cache = any(weight > 0 for weight in cache_weights)
    cache_settings = list(itertools.product(cache_temperatures, decays, minimums, similarities)) if use_cache else [(15., 0., 1, -1.)]
    total_nll = torch.zeros(len(temperatures), len(cache_settings), len(cache_weights), len(count_settings), dtype=torch.float64)
    batches = math.ceil((len(tokens)-1)/(256*args.batch_size))
    target_count = 0
    started = time.perf_counter()
    print(json.dumps(dict(checkpoint=str(args.checkpoint), selection_split='validation', batches=batches,
                          grid_size=total_nll.numel(), device=str(device))), flush=True)
    for batch_index, (ids, targets) in enumerate(windows(tokens, args.batch_size), 1):
        ids, targets = ids.to(device), targets.to(device)
        valid = targets != -100
        hidden, cache_hidden = model.features_for_cache(ids)
        logits = model.head(hidden).float()
        base = target_base_probabilities(logits, targets, temperatures)
        if any(power > 0 for power in confidence_powers):
            full_count_prob, confidence = model.ngram_expert.probabilities_and_confidence(ids)
        else:
            full_count_prob, confidence = model.ngram_expert(ids), None
        count_prob = full_count_prob.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        coefficients = []
        for setting in count_settings:
            coefficient = count_mixing_coefficients(setting, ids, confidence)
            coefficients.append(torch.full_like(count_prob, coefficient) if isinstance(coefficient, float) else coefficient)
        coefficients = torch.stack(coefficients)
        if use_cache:
            cache, active = target_cache_probabilities(ids, cache_hidden, targets, cache_settings)
        else:
            cache = base.new_zeros((1, *ids.shape))
            active = torch.zeros_like(cache, dtype=torch.bool)
        total_nll += accumulate_hybrid_nll(base, cache, active, count_prob, cache_weights,
                                         [s['ngram_weight'] for s in count_settings], valid, coefficients).cpu()
        target_count += int(valid.sum())
        if batch_index % 5 == 0 or batch_index == batches:
            print(json.dumps(dict(batch=batch_index, batches=batches, seconds=time.perf_counter()-started)), flush=True)
    sweep_seconds = time.perf_counter()-started
    rows = []
    for i, temperature in enumerate(temperatures):
        for j, (cache_temperature, cache_decay, minimum, similarity) in enumerate(cache_settings):
            for k, cache_weight in enumerate(cache_weights):
                for m, count_setting in enumerate(count_settings):
                    nll = float(total_nll[i, j, k, m])
                    rows.append(dict(logit_temperature=temperature, cache_temperature=cache_temperature,
                                     cache_layer=args.cache_layer,
                                     cache_decay=cache_decay, cache_min_history=minimum, cache_min_similarity=similarity, cache_weight=cache_weight,
                                     **count_setting, nll_nats=nll, bpb=nll/math.log(2)/byte_count))
    rows.sort(key=lambda row: row['bpb'])
    best = rows[0]
    selected = {name: best[name] for name in ('logit_temperature', 'cache_temperature', 'cache_decay',
                                           'cache_layer',
                                           'cache_min_history', 'cache_min_similarity', 'cache_weight', 'ngram_weight',
                                           'ngram_confidence_power', 'ngram_early_weight', 'ngram_early_tokens')}
    verify_sources_unchanged(source_hashes)
    model.config.update(selected)
    checked = score(model, tokens, byte_count, device, 'fp32')
    checked.pop('window_nll_nats')
    if abs(checked['bpb']-best['bpb']) > 2e-6:
        raise RuntimeError(f'Sweep/scorer discrepancy: {best["bpb"]} vs {checked["bpb"]}')
    verify_sources_unchanged(source_hashes)
    if sha(args.checkpoint) != parent_sha or sha(asset) != asset_sha:
        raise RuntimeError('Parent checkpoint or count asset changed during validation selection.')
    metadata = dict(selection_split='validation', test_used=False, parent_checkpoint=str(args.checkpoint.resolve()),
                    parent_sha256=parent_sha, implementation='hybrid', implementation_sha256=implementation_sha,
                    inference_source_sha256={name: source_hashes[name] for name in ['student.py', 'hybrid.py', 'ngram_expert.py']},
                    tuning_source_sha256=source_hashes, source_stability_verified=True,
                    tuner_sha256=source_hashes['tune_hybrid.py'], ngram_asset=relative_asset, ngram_sha256=asset_sha,
                    selected=selected, validation=checked, targets=target_count, utf8_bytes=byte_count,
                    sweep_seconds=sweep_seconds, grid_size=len(rows), device=str(device), precision='fp32', rows=rows)
    result = dict(checkpoint)
    result['implementation'] = 'hybrid'
    result['config'] = dict(config, **selected)
    result['inference_source_sha256'] = metadata['inference_source_sha256']
    result['postprocessing_selection'] = {key: value for key, value in metadata.items() if key != 'rows'}
    result['validation'] = checked
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output_dir/'checkpoint.pt')
    metadata['checkpoint_sha256'] = sha(args.output_dir/'checkpoint.pt')
    (args.output_dir/'hybrid_sweep.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in metadata.items() if key != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
