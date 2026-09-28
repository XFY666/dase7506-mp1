"""Select a training-only two-neural-model teacher on validation, with no fitting.

The default declared grid is five common temperatures by five mixture weights.
Only aggregate scores are saved. Members and their tied weights are embedded;
the resulting checkpoint needs no external parent path at inference time.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from common import PROTOCOL, ROOT, make_model, setup, sha, windows
from evaluate import score
from train_experiment import cpu_state
from tune_cache import number_list, validation_data


def training_config(config):
    return {name: value for name, value in config.items()
            if not name.startswith(('cache_', 'ngram_')) and name not in ('cpu_batch_chunk', 'logit_temperature')}


def find_run(checkpoint_path):
    for folder in (checkpoint_path.parent, checkpoint_path.parent.parent):
        path = folder/'run.json'
        if path.is_file():
            return path
    raise ValueError(f'Use a direct training checkpoint with a run.json: {checkpoint_path}')


def parent_provenance(path, checkpoint):
    if (checkpoint.get('protocol') != PROTOCOL or checkpoint.get('implementation') != 'student'
            or checkpoint.get('teacher') or checkpoint.get('training_only')
            or int(checkpoint.get('train_tokens', 0)) <= 0):
        raise ValueError('Members must be directly trained student checkpoints, without teacher dependencies.')
    if checkpoint.get('total_training_targets', checkpoint['train_tokens']) != checkpoint['train_tokens']:
        raise ValueError('Unresolved extra training dependency in member checkpoint.')
    run_path = find_run(path)
    metadata = json.loads(run_path.read_text(encoding='utf-8'))
    if (metadata.get('protocol') != PROTOCOL or metadata.get('implementation') != 'student'
            or metadata.get('teacher') or metadata.get('test_used') is not False
            or metadata.get('seed') != checkpoint['seed']
            or metadata.get('config') != checkpoint['config']
            or metadata.get('source_sha256') != checkpoint.get('source_sha256')
            or metadata.get('ancestry', []) != checkpoint.get('ancestry', [])):
        raise ValueError('Checkpoint and source training-run provenance disagree.')
    nodes = list(checkpoint.get('ancestry', [])) + [dict(checkpoint=str(path.resolve()),
                sha256=sha(path), train_tokens=int(checkpoint['train_tokens']))]
    segments = []
    previous_targets = 0
    for node in nodes:
        ancestor_path = Path(node['checkpoint'])
        if not ancestor_path.is_file() or sha(ancestor_path) != node['sha256']:
            raise ValueError(f'Ancestry checkpoint is missing or changed: {ancestor_path}')
        ancestor_run = find_run(ancestor_path)
        run = json.loads(ancestor_run.read_text(encoding='utf-8'))
        run_start = int(run.get('ancestry', [{}])[-1].get('train_tokens', 0)) if run.get('ancestry') else 0
        endpoint = int(node['train_tokens'])
        if (run_start != previous_targets or endpoint < run_start or run.get('teacher')
                or run.get('protocol') != PROTOCOL or run.get('implementation') != 'student'
                or run.get('test_used') is not False):
            raise ValueError('Ambiguous or decreasing training ancestry; supply explicit compatible segments.')
        run_sha = sha(ancestor_run)
        segments.append(dict(segment_id=run_sha, run_json=str(ancestor_run.resolve()),
                             run_json_sha256=run_sha, inherited_targets=run_start,
                             cumulative_endpoint_targets=endpoint, new_targets=endpoint-run_start,
                             reference_checkpoint=str(ancestor_path.resolve()),
                             reference_checkpoint_sha256=node['sha256']))
        previous_targets = endpoint
    return dict(checkpoint=str(path.resolve()), checkpoint_sha256=sha(path),
                seed=checkpoint['seed'], weight_kind=checkpoint.get('weight_kind', 'raw'),
                step=checkpoint['step'], train_tokens=int(checkpoint['train_tokens']),
                config=checkpoint['config'], source_sha256=checkpoint['source_sha256'],
                run_json=str(run_path.resolve()), run_json_sha256=sha(run_path), segments=segments)


def unique_ancestry(parents):
    """Union executed prefixes within each run; resume prefixes appear once."""
    union = {}
    for parent in parents:
        if sum(segment['new_targets'] for segment in parent['segments']) != parent['train_tokens']:
            raise ValueError('Member ancestry segments do not sum to its cumulative targets.')
        for segment in parent['segments']:
            key = segment['segment_id']
            if key in union and union[key]['inherited_targets'] != segment['inherited_targets']:
                raise ValueError('A run segment has conflicting start points.')
            if key not in union or segment['new_targets'] > union[key]['new_targets']:
                union[key] = dict(segment)
    segments = list(union.values())
    return sum(segment['new_targets'] for segment in segments), segments


def target_log_probabilities(logits, targets, temperatures):
    selected = logits.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return torch.stack([selected/t-torch.logsumexp(logits/t, -1) for t in temperatures])


def aggregate_nll(first, second, alphas, valid):
    result = torch.empty(first.shape[0], len(alphas), dtype=torch.float64, device=first.device)
    for index, alpha in enumerate(alphas):
        mixed = second if alpha == 0 else first if alpha == 1 else torch.logaddexp(
            first+math.log(alpha), second+math.log1p(-alpha))
        result[:, index] = (-mixed).masked_fill(~valid[None], 0.).double().sum((-2, -1))
    return result


@torch.no_grad()
def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-checkpoint', required=True, type=Path)
    parser.add_argument('--second-checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--temperatures', default='.95,1,1.05,1.1,1.15')
    parser.add_argument('--alphas', default='0,.25,.5,.75,1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    temperatures, alphas = number_list(args.temperatures), number_list(args.alphas)
    if (any(not math.isfinite(t) or t <= 0 for t in temperatures)
            or any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas) or args.batch_size < 1):
        parser.error('Invalid temperature, mixture weight or batch size.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new empty output directory.')
    paths = [args.first_checkpoint, args.second_checkpoint]
    if paths[0].resolve() == paths[1].resolve():
        parser.error('Use two distinct checkpoints.')
    sources = {name: sha(ROOT/name) for name in ['teacher_ensemble.py', 'student.py',
               'build_teacher_ensemble.py', 'train_experiment.py', 'tune_cache.py', 'common.py', 'evaluate.py']}
    initial_parent_hashes = [sha(path) for path in paths]
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    parents = [parent_provenance(path, checkpoint) for path, checkpoint in zip(paths, checkpoints)]
    if initial_parent_hashes != [parent['checkpoint_sha256'] for parent in parents]:
        raise RuntimeError('Member checkpoint changed while loading.')
    total_targets, segments = unique_ancestry(parents)
    config = dict(vocab=2048, context=256, training_only=True,
                  member_implementations=['student', 'student'],
                  member_configs=[training_config(checkpoint['config']) for checkpoint in checkpoints],
                  common_temperature=1., alpha=.5)
    device, _ = setup(args.device, 'fp32', args.threads)
    model, implementation_sha = make_model('teacher_ensemble', config, device)
    for member, checkpoint in zip(model.members, checkpoints):
        if not torch.equal(checkpoint['model']['token.weight'], checkpoint['model']['head.weight']):
            raise ValueError('Member checkpoint contains inconsistent tied weights.')
        member.load_state_dict(checkpoint['model'], strict=True)
    del checkpoints
    model.eval()
    tokens, byte_count = validation_data()
    total_nll = torch.zeros(len(temperatures), len(alphas), dtype=torch.float64)
    target_count = 0
    started = time.perf_counter()
    for index, (ids, targets) in enumerate(windows(tokens, args.batch_size), 1):
        ids, targets = ids.to(device), targets.to(device)
        first = target_log_probabilities(model.members[0](ids).float(), targets, temperatures)
        second = target_log_probabilities(model.members[1](ids).float(), targets, temperatures)
        total_nll += aggregate_nll(first, second, alphas, targets != -100).cpu()
        target_count += int((targets != -100).sum())
        if index % 5 == 0:
            print(json.dumps(dict(batch=index, seconds=time.perf_counter()-started)), flush=True)
    sweep_seconds = time.perf_counter()-started
    rows = [dict(common_temperature=t, alpha=a, nll_nats=float(total_nll[i,j]),
                 bpb=float(total_nll[i,j])/math.log(2)/byte_count)
            for i,t in enumerate(temperatures) for j,a in enumerate(alphas)]
    rows.sort(key=lambda row: row['bpb'])
    best = rows[0]
    model.config.update(common_temperature=best['common_temperature'], alpha=best['alpha'])

    def assert_stable():
        if sources != {name: sha(ROOT/name) for name in sources}:
            raise RuntimeError('Teacher source changed during selection.')
        for parent in parents:
            if sha(Path(parent['checkpoint'])) != parent['checkpoint_sha256']:
                raise RuntimeError('Member checkpoint changed during selection.')
            for segment in parent['segments']:
                if sha(Path(segment['run_json'])) != segment['run_json_sha256']:
                    raise RuntimeError('Member run provenance changed during selection.')
                if sha(Path(segment['reference_checkpoint'])) != segment['reference_checkpoint_sha256']:
                    raise RuntimeError('Member ancestry checkpoint changed during selection.')

    assert_stable()
    checked = score(model, tokens, byte_count, device, 'fp32')
    checked.pop('window_nll_nats')
    if abs(checked['bpb']-best['bpb']) > 2e-6:
        raise RuntimeError('Ensemble aggregate/scorer discrepancy.')
    assert_stable()
    endpoints = [row for row in rows if row['alpha'] in (0., 1.)]
    metadata = dict(protocol=PROTOCOL, training_only=True, needed_for_final_inference=False,
                    final_submission_allowed=False, selection_split='validation', test_used=False,
                    selected=dict(common_temperature=best['common_temperature'], alpha=best['alpha']),
                    validation=checked, grid=dict(temperatures=temperatures, alphas=alphas), rows=rows,
                    improvement_over_best_endpoint_bpb=min(row['bpb'] for row in endpoints)-best['bpb'] if endpoints else None,
                    new_training_targets=0, unique_teacher_ancestry_targets=total_targets,
                    ancestry_segments=segments, parent_checkpoints=parents,
                    ancestry_rule='Union executed run prefixes by run.json SHA; each resumed prefix counted once.',
                    dependency_scope='All parents used in selection, conservatively including inactive endpoint members.',
                    source_sha256=sources, source_stability_verified=True,
                    sweep_seconds=sweep_seconds, targets=target_count, utf8_bytes=byte_count,
                    device=str(device), precision='fp32', kd_temperature_supported=1.,
                    member_forward_count=1 if best['alpha'] in (0., 1.) else 2)
    checkpoint = dict(protocol=PROTOCOL, implementation='teacher_ensemble', config=dict(model.config),
                      model=cpu_state(model), seed=None, weight_kind='probability_ensemble',
                      train_tokens=0, total_training_targets=total_targets, ancestry=segments,
                      training_only=True, final_submission_allowed=False, kd_temperature_supported=1.,
                      source_sha256=sources,
                      inference_source_sha256={name:sources[name] for name in ['teacher_ensemble.py', 'student.py']},
                      validation=checked, ensemble_selection={k:v for k,v in metadata.items() if k != 'rows'})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_dir/'checkpoint.pt')
    metadata['checkpoint_sha256'] = sha(args.output_dir/'checkpoint.pt')
    metadata['process_seconds'] = time.perf_counter()-process_started
    metadata['process_timer_scope'] = 'Builder main through checkpoint save; excludes Python import/startup.'
    (args.output_dir/'ensemble_sweep.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('rows','parent_checkpoints')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
