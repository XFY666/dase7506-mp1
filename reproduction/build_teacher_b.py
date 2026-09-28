"""Build teacher B, the second-generation distillation teacher.

Teacher B mixes the probabilities of two trained students at a shared
temperature T:

    q = alpha * softmax(z_1 / T) + (1 - alpha) * softmax(z_2 / T)

The first member is the directly trained 6x288 model and the second is the
8x256 student distilled from teacher A; the recipe uses T = 1.05 and
alpha = 0.25. No parameters are fitted. The output is a training-only target
generator for train_experiment.py and is never used for inference.

build_teacher_ensemble.py only accepts directly trained members, so this
script is needed for a distilled member. Run it from the repository root:

    python reproduction/build_teacher_b.py --first-checkpoint RUNS/direct-6x288/checkpoint.pt \
        --second-checkpoint RUNS/kd-8x256/checkpoint.pt --output-dir RUNS/teacher-b
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from build_teacher_ensemble import find_run, training_config  # noqa: E402
from common import PROTOCOL, make_model, setup, sha  # noqa: E402
from evaluate import score  # noqa: E402
from train_experiment import cpu_state  # noqa: E402
from tune_cache import validation_data  # noqa: E402


def own_segments(path, checkpoint):
    """Training segments executed by the run(s) that produced this checkpoint."""
    nodes = list(checkpoint.get('ancestry', [])) + [
        dict(checkpoint=str(path.resolve()), sha256=sha(path), train_tokens=int(checkpoint['train_tokens']))]
    segments, previous = [], 0
    for node in nodes:
        node_path = Path(node['checkpoint'])
        run = find_run(node_path)
        run_sha = sha(run)
        endpoint = int(node['train_tokens'])
        segments.append(dict(segment_id=run_sha, run_json=str(run.resolve()), run_json_sha256=run_sha,
                             inherited_targets=previous, cumulative_endpoint_targets=endpoint,
                             new_targets=endpoint-previous, reference_checkpoint=str(node_path.resolve()),
                             reference_checkpoint_sha256=node['sha256']))
        previous = endpoint
    return segments


def member_segments(path, checkpoint):
    if checkpoint.get('protocol') != PROTOCOL or checkpoint.get('implementation') != 'student':
        raise ValueError(f'Members must be student checkpoints: {path}')
    segments = own_segments(path, checkpoint)
    teacher = checkpoint.get('teacher')
    if teacher:
        # A distilled member also depends on everything its own teacher was trained on.
        segments = list(teacher.get('ancestry', [])) + segments
    return segments


def union(segments):
    unique = {}
    for segment in segments:
        key = segment['segment_id']
        if key not in unique or segment['new_targets'] > unique[key]['new_targets']:
            unique[key] = dict(segment)
    return sum(item['new_targets'] for item in unique.values()), list(unique.values())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--first-checkpoint', type=Path, required=True, help='directly trained 6x288 model')
    parser.add_argument('--second-checkpoint', type=Path, required=True, help='student distilled from teacher A')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--temperature', type=float, default=1.05)
    parser.add_argument('--alpha', type=float, default=.25, help='probability weight of the first member')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--skip-validation', action='store_true',
                        help='do not score the teacher on the validation split')
    args = parser.parse_args()
    if not (math.isfinite(args.temperature) and args.temperature > 0 and 0 < args.alpha < 1):
        parser.error('Use a positive temperature and 0 < alpha < 1.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new empty output directory.')
    paths = [args.first_checkpoint, args.second_checkpoint]
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    total_targets, segments = union([s for p, c in zip(paths, checkpoints) for s in member_segments(p, c)])

    config = dict(vocab=2048, context=256, training_only=True,
                  member_implementations=['student', 'student'],
                  member_configs=[training_config(c['config']) for c in checkpoints],
                  common_temperature=args.temperature, alpha=args.alpha)
    device, _ = setup(args.device, 'fp32', args.threads)
    model, _ = make_model('teacher_ensemble', config, device)
    for member, checkpoint in zip(model.members, checkpoints):
        member.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    validation = None
    if not args.skip_validation:
        tokens, byte_count = validation_data()
        validation = score(model, tokens, byte_count, device, 'fp32')
        validation.pop('window_nll_nats')

    sources = {name: sha(ROOT/name) for name in ('teacher_ensemble.py', 'student.py', 'train_experiment.py')}
    parents = [dict(checkpoint=str(p.resolve()), checkpoint_sha256=sha(p), config=c['config'],
                    step=c.get('step'), weight_kind=c.get('weight_kind'), train_tokens=int(c['train_tokens']))
               for p, c in zip(paths, checkpoints)]
    selection = dict(protocol=PROTOCOL, training_only=True, final_submission_allowed=False,
                     selected=dict(common_temperature=args.temperature, alpha=args.alpha),
                     selection_split='validation', test_used=False, validation=validation,
                     new_training_targets=0, unique_teacher_ancestry_targets=total_targets,
                     parent_checkpoints=parents, kd_temperature_supported=1., member_forward_count=2,
                     builder_sha256=sha(Path(__file__)))
    checkpoint = dict(protocol=PROTOCOL, implementation='teacher_ensemble', config=dict(model.config),
                      model=cpu_state(model), seed=None, weight_kind='probability_ensemble',
                      train_tokens=0, total_training_targets=total_targets, ancestry=segments,
                      training_only=True, final_submission_allowed=False, kd_temperature_supported=1.,
                      source_sha256=sources,
                      inference_source_sha256={k: sources[k] for k in ('teacher_ensemble.py', 'student.py')},
                      validation=validation, ensemble_selection=selection)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_dir/'checkpoint.pt')
    summary = dict(checkpoint=str(args.output_dir/'checkpoint.pt'), checkpoint_sha256=sha(args.output_dir/'checkpoint.pt'),
                   common_temperature=args.temperature, alpha=args.alpha,
                   unique_teacher_ancestry_targets=total_targets,
                   validation_bpb=None if validation is None else validation['bpb'])
    (args.output_dir/'teacher_b.json').write_text(json.dumps(dict(summary, parents=parents), indent=2)+'\n',
                                                  encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
