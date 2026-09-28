"""Select arithmetic averages of raw checkpoints using validation only.

Automatic mode builds averages near the best raw snapshot and at the trajectory
tail. Explicit mode averages the supplied paths. All parents must belong to the
same recorded training run; that run may itself resume a recorded ancestor.
The original scorer assesses every candidate in FP32. No test text is loaded.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from common import PROTOCOL, make_model, setup, sha
from evaluate import score
from tune_cache import validation_data


def training_config(config):
    """Ignore inference-only scalars when checking the training architecture."""
    return {key: value for key, value in config.items()
            if not key.startswith('cache_') and key != 'logit_temperature'}


def run_directory(path):
    for directory in (path.parent, *list(path.parents)[1:4]):
        if (directory / 'run.json').is_file():
            return directory.resolve()
    raise ValueError(f'Cannot verify the training run for {path}; run.json is required nearby.')


def load_parents(paths):
    """Require raw snapshots of one recorded optimization trajectory."""
    result = []
    for path in paths:
        path = path.resolve()
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('protocol') != PROTOCOL:
            raise ValueError(f'Unexpected checkpoint protocol: {path}')
        if checkpoint.get('weight_kind') != 'raw':
            raise ValueError(f'Only raw training snapshots can be averaged: {path}')
        if 'step' not in checkpoint or 'train_tokens' not in checkpoint:
            raise ValueError(f'Missing step or cumulative training-target count: {path}')
        owner = run_directory(path)
        run_metadata = json.loads((owner / 'run.json').read_text())
        if (checkpoint['seed'] != run_metadata['seed'] or
                checkpoint['implementation'] != run_metadata['implementation'] or
                training_config(checkpoint['config']) != training_config(run_metadata['config'])):
            raise ValueError(f'Checkpoint does not match its run.json: {path}')
        info = dict(path=str(path), sha256=sha(path), step=int(checkpoint['step']),
                    train_tokens=int(checkpoint['train_tokens']), weight_kind='raw')
        if checkpoint.get('validation'):
            info['recorded_validation_bpb'] = float(checkpoint['validation']['bpb'])
        result.append(dict(checkpoint=checkpoint, info=info, run_directory=owner,
                           run_metadata=run_metadata))
    if not result:
        raise ValueError('No snapshots were supplied.')
    reference = result[0]
    for item in result[1:]:
        first, other = reference['checkpoint'], item['checkpoint']
        if item['run_directory'] != reference['run_directory']:
            raise ValueError('All parents must belong to the same recorded run, not unrelated models.')
        for key in ('implementation', 'seed', 'ancestry', 'source_sha256'):
            if first.get(key) != other.get(key):
                raise ValueError(f'Incompatible checkpoint provenance field: {key}')
        if training_config(first['config']) != training_config(other['config']):
            raise ValueError('The checkpoint training configurations differ.')
    result.sort(key=lambda item: item['info']['step'])
    if len({item['info']['step'] for item in result}) != len(result):
        raise ValueError('Supply one raw snapshot per optimization step.')
    if any(later['info']['train_tokens'] < earlier['info']['train_tokens']
           for earlier, later in zip(result, result[1:])):
        raise ValueError('Cumulative training-target counts must not decrease with step.')
    return result


def alias_groups(reference_state):
    grouped = {}
    for name, value in reference_state.items():
        identity = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
        grouped.setdefault(identity, []).append(name)
    return list(grouped.values())


def average_states(states, reference_state):
    """Average floating tensors in FP32, preserve exact buffers and tied storage."""
    if not states:
        raise ValueError('At least one state dictionary is required.')
    expected = set(reference_state)
    for state in states:
        if set(state) != expected:
            raise ValueError('Checkpoint state dictionary keys differ.')
        for name, reference in reference_state.items():
            if state[name].shape != reference.shape:
                raise ValueError(f'Checkpoint tensor shape mismatch: {name}')
            if state[name].is_floating_point() != reference.is_floating_point():
                raise ValueError(f'Checkpoint tensor type mismatch: {name}')
    result = {}
    for names in alias_groups(reference_state):
        name = names[0]
        tensors = [state[name].detach().cpu() for state in states]
        for state in states:
            for alias in names[1:]:
                if not torch.equal(state[name], state[alias]):
                    raise ValueError(f'Inconsistent tied parameters: {name}, {alias}')
        if tensors[0].is_floating_point():
            averaged = torch.zeros_like(tensors[0], dtype=torch.float32)
            for tensor in tensors:
                if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                    raise ValueError(f'Nonfinite or incompatible floating tensor: {name}')
                averaged.add_(tensor.float())
            averaged.div_(len(tensors))
        else:
            if any(tensor.dtype != tensors[0].dtype or not torch.equal(tensor, tensors[0])
                   for tensor in tensors[1:]):
                raise ValueError(f'Nonfloating buffer changed: {name}')
            averaged = tensors[0].clone()
        for alias in names:
            result[alias] = averaged
    return result


def candidate_groups(parents, counts, explicit=False):
    """Return distinct groups of parent indices, including the best raw anchor."""
    measured = [i for i, item in enumerate(parents) if 'recorded_validation_bpb' in item['info']]
    if measured:
        best = min(measured, key=lambda i: parents[i]['info']['recorded_validation_bpb'])
    elif explicit:
        best = len(parents) - 1
    else:
        raise ValueError('Automatic selection requires recorded validation BPB for snapshots.')
    groups = [('best_raw', [best])]
    if explicit:
        if len(parents) < 2:
            raise ValueError('Explicit averaging requires at least two raw snapshots.')
        groups.append(('explicit_mean', list(range(len(parents)))))
        return groups
    seen = {(best,)}
    for count in counts:
        if count < 2:
            raise ValueError('Average sizes must be at least two.')
        if count > len(parents):
            continue
        windows = [('before_best', max(0, best - count + 1)),
                   ('near_best', max(0, min(len(parents) - count, best - count // 2))),
                   ('tail', len(parents) - count)]
        for label, start in windows:
            # A before-best average cannot include later checkpoints.
            if label == 'before_best' and best + 1 < count:
                continue
            group = tuple(range(start, start + count))
            if group not in seen:
                groups.append((f'{label}_{count}', list(group)))
                seen.add(group)
    return groups


def make_average(parents, reference_state, script_hash):
    latest = max(parents, key=lambda item: item['info']['step'])
    checkpoint = copy.copy(latest['checkpoint'])
    for key in ('optimizer', 'ema', 'sampling_rng', 'torch_rng', 'cuda_rng',
                'postprocessing_selection', 'averaging_selection'):
        checkpoint.pop(key, None)
    checkpoint['config'] = dict(training_config(checkpoint['config']),
                                cache_weight=0., logit_temperature=1.)
    checkpoint['model'] = average_states([item['checkpoint']['model'] for item in parents], reference_state)
    inherited = max(int(item['checkpoint']['train_tokens']) for item in parents)
    for item in parents:
        inherited = max([inherited] + [int(ancestor.get('train_tokens', 0))
                                      for ancestor in item['checkpoint'].get('ancestry', [])])
    checkpoint['train_tokens'] = inherited
    checkpoint['weight_kind'] = 'arithmetic_mean' if len(parents) > 1 else 'raw'
    checkpoint['validation'] = None
    checkpoint['averaging'] = dict(
        method='arithmetic_mean', parent_checkpoints=[item['info'] for item in parents],
        parent_run=str(latest['run_directory']), inherited_train_targets=inherited,
        additional_train_targets=0, cost_rule='maximum cumulative ancestor cost within one trajectory',
        new_weight_fitting=False, script_sha256=script_hash,
        parent_ancestry=copy.deepcopy(latest['checkpoint'].get('ancestry', [])),
    )
    return checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--checkpoints', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--counts', default='2,4,8')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new empty output directory.')
    paths = sorted((args.run_dir / 'snapshots').glob('step*.pt')) if args.run_dir else args.checkpoints
    parents = load_parents(paths)
    groups = candidate_groups(parents, [int(item) for item in args.counts.split(',')],
                              explicit=args.checkpoints is not None)
    device, _ = setup(args.device, 'fp32', args.threads)
    template_config = dict(training_config(parents[0]['checkpoint']['config']),
                           cache_weight=0., logit_temperature=1.)
    model, implementation_sha = make_model(parents[0]['checkpoint']['implementation'],
                                           template_config, torch.device('cpu'))
    reference_state = model.state_dict()
    model.to(device)
    model.eval()
    tokens, byte_count = validation_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = args.output_dir / 'candidates'
    candidates_dir.mkdir()
    script_hash = sha(Path(__file__))
    rows = []
    started = time.perf_counter()
    best = None
    print(json.dumps({'started': str(args.output_dir), 'candidates': len(groups),
                      'selection_split': 'validation', 'device': str(device)}), flush=True)
    for name, indices in groups:
        selected_parents = [parents[index] for index in indices]
        averaged = make_average(selected_parents, reference_state, script_hash)
        destination = candidates_dir / f'{name}.pt'
        torch.save(averaged, destination)
        model.load_state_dict(averaged['model'])
        measured = score(model, tokens, byte_count, device, 'fp32')
        measured.pop('window_nll_nats')
        averaged['validation'] = measured
        averaged['averaging']['validation'] = measured
        torch.save(averaged, destination)
        row = dict(name=name, checkpoint=str(destination.resolve()), sha256=sha(destination),
                   steps=[item['info']['step'] for item in selected_parents],
                   train_tokens=averaged['train_tokens'], **measured)
        rows.append(row)
        if best is None or row['bpb'] < best['bpb']:
            best = row
        print(json.dumps(row), flush=True)
    # Search cost counts the training trajectory once, including unused snapshots.
    search_training_targets = max(item['info']['train_tokens'] for item in parents)
    metrics_path = parents[0]['run_directory'] / 'metrics.json'
    if metrics_path.is_file():
        search_training_targets = max(search_training_targets,
                                      int(json.loads(metrics_path.read_text()).get('train_tokens', 0)))
    summary = dict(selection_split='validation', test_used=False, precision='fp32', device=str(device),
                   source_run=str(parents[0]['run_directory']), script_sha256=script_hash,
                   implementation_sha256=implementation_sha, candidates=rows, selected=best['name'],
                   best_validation_bpb=best['bpb'], inherited_search_train_targets=search_training_targets,
                   additional_train_targets=0, seconds=time.perf_counter() - started)
    winner = torch.load(best['checkpoint'], map_location='cpu', weights_only=True)
    winner['averaging_selection'] = {key: value for key, value in summary.items() if key != 'candidates'}
    torch.save(winner, args.output_dir / 'checkpoint.pt')
    summary['checkpoint_sha256'] = sha(args.output_dir / 'checkpoint.pt')
    (args.output_dir / 'averaging_results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
