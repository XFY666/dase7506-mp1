"""Select temperature/cache scalars on validation; save only aggregate evidence.

The model weights and features are fixed. No per-token probabilities, hidden
states, or validation cache are written to disk. The selected checkpoint is
cross-checked with the unchanged official scorer on the validation split.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from common import PROTOCOL, ROOT, make_model, setup, sha, windows
from evaluate import score
from student import causal_cache_max_similarity, neural_cache_distribution


def number_list(text, converter=float):
    return [converter(item) for item in text.split(',')]


def assert_sources_unchanged(expected):
    """Refuse to label a sweep with source that changed while it was running."""
    changed = [name for name, digest in expected.items() if sha(ROOT / name) != digest]
    if changed:
        raise RuntimeError('Source changed during cache tuning; rerun with stable files: ' + ', '.join(changed))


def validation_data():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    tokenizer_path = ROOT / 'data/tokenizer.json'
    text_path = ROOT / 'data/wikitext_validation.txt'
    for path in (tokenizer_path, text_path):
        if sha(path) != manifest['sha256'][path.name]:
            raise ValueError(f'Changed benchmark file: {path.name}')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw = text_path.read_bytes()
    return torch.tensor(tokenizer.encode(raw.decode('utf8')).ids, dtype=torch.long), len(raw)


def target_base_probabilities(logits, targets, temperatures):
    """An ephemeral [temperature,batch,time] tensor; never an inference asset."""
    selected = logits.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return torch.stack([(selected / temperature - torch.logsumexp(logits / temperature, -1)).exp()
                        for temperature in temperatures])


def target_cache_probabilities(ids, hidden, targets, cache_settings):
    """Compute only probabilities of reference tokens for development metrics.

    References are used here by the tuning evaluator to assess the predictor.
    They do not enter neural_cache_distribution or the submitted predictor.
    Settings may be (temperature, decay, minimum history) or those three values
    followed by a minimum cosine similarity. Omitted similarity disables gating.
    """
    values_match_target = ids[:, None, 1:] == targets[:, :, None]
    normalized = F.normalize(hidden.float(), dim=-1)
    similarity = torch.matmul(normalized, normalized[:, :-1].transpose(-1, -2))
    answers = []
    active_rows = []
    computed = {}
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    maximum_similarity = None
    for setting in cache_settings:
        if len(setting) == 3:
            temperature, decay, minimum = setting
            minimum_similarity = -1.
        elif len(setting) == 4:
            temperature, decay, minimum, minimum_similarity = setting
        else:
            raise ValueError('Cache settings must contain three or four values.')
        if not -1. <= minimum_similarity <= 1.:
            raise ValueError('cache_min_similarity must be in [-1, 1].')
        key = (temperature, decay)
        if key not in computed:
            weights, _ = neural_cache_distribution(ids, hidden, temperature, decay, 1, similarity)
            computed[key] = (weights * values_match_target).sum(-1)
        answers.append(computed[key])
        active = (positions >= max(1, minimum)).expand(ids.shape[0], -1)
        if minimum_similarity > -1.:
            if maximum_similarity is None:
                maximum_similarity = causal_cache_max_similarity(similarity)
            active = active & (maximum_similarity >= minimum_similarity)
        active_rows.append(active)
    return torch.stack(answers), torch.stack(active_rows)


def accumulate_mixture_nll(base_prob, cache_prob, active, mixing_weights, valid):
    """Return aggregate NLL [logit temperature,cache setting,mixing weight]."""
    result = torch.empty(base_prob.shape[0], cache_prob.shape[0], len(mixing_weights),
                         device=base_prob.device, dtype=torch.float64)
    # One coefficient at a time bounds temporary memory independently of grid size.
    for index, weight in enumerate(mixing_weights):
        coefficient = active.float() * weight
        mixed = base_prob[:, None] * (1 - coefficient[None]) + cache_prob[None] * coefficient[None]
        losses = -mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()
        losses = losses.masked_fill(~valid[None, None], 0.)
        result[:, :, index] = losses.double().sum((-2, -1))
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', default=4, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--logit-temperatures', default='.9,.95,1,1.05,1.1')
    parser.add_argument('--cache-temperatures', default='8,12,16,24,32,48')
    parser.add_argument('--weights', default='0,.025,.05,.075,.1,.15,.2')
    parser.add_argument('--decays', default='0,.01')
    parser.add_argument('--min-histories', default='1,8')
    parser.add_argument('--min-similarities', default='-1',
                        help='Comma-separated cosine gate thresholds; -1 disables gating.')
    parser.add_argument('--cache-layer', type=int, default=None,
                        help='-1 uses final normalized features; 1..depth uses a raw block output.')
    args = parser.parse_args()
    temperatures = number_list(args.logit_temperatures)
    cache_temperatures = number_list(args.cache_temperatures)
    decays = number_list(args.decays)
    minimums = number_list(args.min_histories, int)
    similarities = number_list(args.min_similarities)
    strengths = number_list(args.weights)
    if any(v <= 0. for v in temperatures) or any(v < 0. for v in cache_temperatures):
        parser.error('Logit temperatures must be positive; cache temperatures must be nonnegative.')
    if any(v < 0. for v in decays) or any(v < 1 for v in minimums):
        parser.error('Decay must be nonnegative and minimum history must be positive.')
    if any(not -1. <= v <= 1. for v in similarities):
        parser.error('Minimum similarities must be in [-1,1].')
    if any(not 0. <= v < 1. for v in strengths) or args.batch_size < 1:
        parser.error('Weights must be in [0,1) and batch size must be positive.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new empty output directory.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device, _ = setup(args.device, 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Unexpected checkpoint protocol.')
    config = dict(checkpoint['config'])
    cache_layer = int(config.get('cache_layer', -1) if args.cache_layer is None else args.cache_layer)
    if cache_layer != -1 and not 1 <= cache_layer <= config['depth']:
        parser.error('cache-layer must be -1 or a block number from 1 through depth.')
    config.update(cache_weight=0., logit_temperature=1., cache_layer=cache_layer)
    initial_sources = {name: sha(ROOT / name) for name in ('student.py', Path(__file__).name)}
    model, implementation_sha = make_model(checkpoint['implementation'], config, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    tokens, byte_count = validation_data()
    cache_settings = list(itertools.product(cache_temperatures, decays, minimums, similarities))
    total_nll = torch.zeros(len(temperatures), len(cache_settings), len(strengths), dtype=torch.float64)
    target_count = 0
    started = time.perf_counter()
    batches = math.ceil((len(tokens) - 1) / (256 * args.batch_size))
    print(json.dumps({'started': str(args.checkpoint), 'batches': batches,
                      'grid_size': total_nll.numel(), 'selection_split': 'validation'}), flush=True)
    for batch_index, (ids, targets) in enumerate(windows(tokens, args.batch_size), 1):
        ids, targets = ids.to(device), targets.to(device)
        valid = targets != -100
        hidden, cache_hidden = model.features_for_cache(ids)
        logits = model.head(hidden).float()
        base = target_base_probabilities(logits, targets, temperatures)
        cache, active = target_cache_probabilities(ids, cache_hidden, targets, cache_settings)
        total_nll += accumulate_mixture_nll(base, cache, active, strengths, valid).cpu()
        target_count += int(valid.sum())
        if batch_index % 5 == 0 or batch_index == batches:
            print(json.dumps({'batch': batch_index, 'batches': batches,
                              'seconds': time.perf_counter() - started}), flush=True)
    sweep_seconds = time.perf_counter() - started
    rows = []
    for logit_index, logit_temperature in enumerate(temperatures):
        for cache_index, (cache_temperature, cache_decay, minimum, minimum_similarity) in enumerate(cache_settings):
            for strength_index, weight in enumerate(strengths):
                nll = float(total_nll[logit_index, cache_index, strength_index])
                rows.append(dict(logit_temperature=logit_temperature,
                                 cache_temperature=cache_temperature, cache_decay=cache_decay,
                                 cache_min_history=minimum, cache_min_similarity=minimum_similarity, cache_weight=weight,
                                 cache_layer=cache_layer,
                                 nll_nats=nll, bpb=nll / math.log(2) / byte_count))
    rows.sort(key=lambda row: row['bpb'])
    best = rows[0]
    selected = {name: best[name] for name in ('logit_temperature', 'cache_temperature',
                'cache_decay', 'cache_min_history', 'cache_min_similarity', 'cache_weight', 'cache_layer')}
    model.config.update(selected)
    assert_sources_unchanged(initial_sources)
    checked = score(model, tokens, byte_count, device, 'fp32')
    checked.pop('window_nll_nats')
    if abs(checked['bpb'] - best['bpb']) > 2e-6:
        raise RuntimeError(f'Sweep/scorer discrepancy: {best["bpb"]} versus {checked["bpb"]}')
    metadata = dict(selection_split='validation', test_used=False, parent_checkpoint=str(args.checkpoint.resolve()),
                    parent_sha256=sha(args.checkpoint), implementation_sha256=implementation_sha,
                    inference_source_sha256={'student.py': initial_sources['student.py']},
                    tuner_sha256=initial_sources[Path(__file__).name], source_sha256_at_start=initial_sources,
                    selected=selected, validation=checked,
                    targets=target_count, utf8_bytes=byte_count, sweep_seconds=sweep_seconds,
                    grid_size=len(rows), device=str(device), precision='fp32', rows=rows)
    # Preserve training provenance and weights; only inference scalars change.
    result = dict(checkpoint)
    result['config'] = dict(config, **selected)
    result['postprocessing_selection'] = {key: value for key, value in metadata.items() if key != 'rows'}
    result['validation'] = checked
    assert_sources_unchanged(initial_sources)
    torch.save(result, args.output_dir / 'checkpoint.pt')
    metadata['checkpoint_sha256'] = sha(args.output_dir / 'checkpoint.pt')
    (args.output_dir / 'cache_sweep.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({key: value for key, value in metadata.items() if key != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
