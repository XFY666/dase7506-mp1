"""Validation-only training/search harness. The supplied scorer is unchanged.

All model weights start randomly unless an explicit, fully recorded resume path
is supplied. Development loads only train and validation text. Checkpoints can
be evaluated directly with the original evaluate.py without retraining.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from common import PROTOCOL, ROOT, autocast, make_model, setup, sha
from evaluate import score


def development_data():
    """Never tokenize or return the test split during model development."""
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    tokenizer_path = ROOT / 'data/tokenizer.json'
    assert sha(tokenizer_path) == manifest['sha256']['tokenizer.json']
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result = {}
    for split in ('train', 'validation'):
        filename = f'wikitext_{split}.txt'
        path = ROOT / 'data' / filename
        assert sha(path) == manifest['sha256'][filename]
        raw = path.read_bytes()
        result[split] = (torch.tensor(tokenizer.encode(raw.decode('utf8')).ids), len(raw))
    return result


def cpu_state(model):
    """Retain tied storage in the portable state dictionary."""
    result, copies = {}, {}
    for name, value in model.state_dict().items():
        key = (value.data_ptr(), tuple(value.shape), value.dtype)
        if key not in copies:
            copies[key] = value.detach().cpu().clone()
        result[name] = copies[key]
    return result


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def token_distillation_kl(student_logits, teacher_logits, temperature):
    """Teacher KL averaged per token; the frozen teacher receives no gradient."""
    if temperature <= 0:
        raise ValueError('Distillation temperature must be positive.')
    return F.kl_div(F.log_softmax(student_logits.float()/temperature, dim=-1),
                    F.softmax(teacher_logits.detach().float()/temperature, dim=-1),
                    reduction='none').sum(-1).mean() * temperature**2


def validate_teacher_checkpoint(checkpoint, temperature):
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] not in ('student', 'model', 'teacher_ensemble'):
        raise ValueError('The teacher must be a compatible, locally trained neural checkpoint.')
    if checkpoint['implementation'] == 'teacher_ensemble':
        if (temperature != 1. or checkpoint.get('kd_temperature_supported') != 1.
                or not checkpoint.get('training_only') or checkpoint.get('final_submission_allowed') is not False):
            raise ValueError('This training-only ensemble teacher supports T_KD=1 only.')
        expected = checkpoint.get('inference_source_sha256', {})
        for name in ('teacher_ensemble.py', 'student.py'):
            if expected.get(name) != sha(ROOT/name):
                raise ValueError(f'Ensemble teacher implementation changed: {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['fp32', 'bf16', 'auto'], default='bf16')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=16000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--min-lr-ratio', type=float, default=.05)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--beta2', type=float, default=.95)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--ema-decay', type=float, default=.999)
    p.add_argument('--ema-start', type=int, default=1000)
    p.add_argument('--patience', type=int, default=0,
                   help='Optional validation events without improvement before early stopping.')
    p.add_argument('--resume', type=Path)
    p.add_argument('--teacher-checkpoint', type=Path,
                   help='Optional teacher trained only on supplied training text; never needed for inference.')
    p.add_argument('--distill-weight', type=float, default=.5)
    p.add_argument('--distill-temperature', type=float, default=1.)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--log-every', type=int, default=100)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) < 1:
        p.error('Steps, batch size and event intervals must be positive.')
    if not 0 <= args.distill_weight <= 1 or args.distill_temperature <= 0:
        p.error('Distillation weight must be in [0,1] and temperature must be positive.')
    if args.teacher_checkpoint and args.resume:
        p.error('Run distillation from scratch so its paired control and ancestry remain explicit.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Use a new, empty run directory, including when resuming.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    process_started = time.perf_counter()
    device, precision = setup(args.device, args.precision, args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    if config.get('cache_weight', 0):
        p.error('Train with cache_weight=0; select any cache settings on validation later.')
    data = development_data()
    model, implementation_sha = make_model(args.implementation, config, device)
    teacher, teacher_info = None, None
    if args.teacher_checkpoint:
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=True)
        validate_teacher_checkpoint(teacher_checkpoint, args.distill_temperature)
        # Creating a teacher must not change the student's paired dropout RNG.
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
            teacher, teacher_sha = make_model(teacher_checkpoint['implementation'], teacher_checkpoint['config'], device)
            teacher.load_state_dict(teacher_checkpoint['model'])
            teacher.eval().requires_grad_(False)
        teacher_info = {
            'checkpoint': str(args.teacher_checkpoint.resolve()), 'sha256': sha(args.teacher_checkpoint),
            'implementation_sha256_at_loading': teacher_sha,
            'implementation': teacher_checkpoint['implementation'], 'config': teacher_checkpoint['config'],
            'seed': teacher_checkpoint['seed'], 'weight_kind': teacher_checkpoint.get('weight_kind', 'raw'),
            'train_tokens': teacher_checkpoint['train_tokens'],
            'total_dependency_train_tokens': teacher_checkpoint.get('total_training_targets', teacher_checkpoint['train_tokens']),
            'ancestry': teacher_checkpoint.get('ancestry', []),
            'inference_source_sha256': teacher_checkpoint.get('inference_source_sha256', {}),
            'ensemble_selection': teacher_checkpoint.get('ensemble_selection'),
            'member_forward_count': teacher_checkpoint.get('ensemble_selection', {}).get('member_forward_count', 1),
            'forward_time_accounting': 'Included in measured student training seconds; not timed separately.',
            'distill_weight': args.distill_weight, 'temperature': args.distill_temperature,
            'targets_source': 'supplied training windows only', 'needed_for_inference': False,
        }
        del teacher_checkpoint
    params_decay = [v for v in model.parameters() if v.ndim >= 2]
    params_other = [v for v in model.parameters() if v.ndim < 2]
    optimizer = torch.optim.AdamW([
        {'params': params_decay, 'weight_decay': args.weight_decay},
        {'params': params_other, 'weight_decay': 0.},
    ], lr=args.lr, betas=(.9, args.beta2), fused=device.type == 'cuda')
    ema = copy.deepcopy(model).eval() if args.ema_decay > 0 else None
    if ema is not None:
        ema.requires_grad_(False)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    start_step, inherited_targets, ancestry = 0, 0, []
    if args.resume:
        resume = torch.load(args.resume, map_location='cpu', weights_only=False)
        if resume.get('teacher'):
            raise ValueError('Resuming a distillation run is unsupported; preserve its teacher dependency and start a fresh recorded run.')
        if resume['config'] != config or resume['implementation'] != args.implementation:
            raise ValueError('Resume architecture does not match.')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        for group in optimizer.param_groups:
            group['betas'] = (.9, args.beta2)
            group['weight_decay'] = args.weight_decay if group is optimizer.param_groups[0] else 0.
        if ema is not None:
            ema.load_state_dict(resume['ema'] if resume['ema'] is not None else resume['model'])
        rng.set_state(resume['sampling_rng'])
        torch.set_rng_state(resume['torch_rng'])
        if device.type == 'cuda' and resume.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        start_step = resume['step']
        inherited_targets = resume['train_tokens']
        ancestry = resume.get('ancestry', []) + [{'checkpoint': str(args.resume.resolve()),
                    'sha256': sha(args.resume), 'train_tokens': inherited_targets}]
        if args.steps <= start_step:
            raise ValueError('--steps is the total step endpoint and must exceed resume step.')
    source_hashes = {str(f.relative_to(ROOT)): sha(f) for f in ROOT.glob('*.py')}
    (args.run_dir / 'source').mkdir()
    for source_name in source_hashes:
        shutil.copy2(ROOT / source_name, args.run_dir / 'source' / source_name)
    shutil.copy2(args.config, args.run_dir / 'source' / 'config.json')
    (args.run_dir / 'snapshots').mkdir()
    metadata = {
        'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
        'seed': args.seed, 'precision': precision,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'parameters': sum(v.numel() for v in model.parameters()),
        'torch_version': str(torch.__version__), 'python_version': platform.python_version(),
        'platform': platform.platform(), 'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        'source_sha256': source_hashes, 'implementation_sha256': implementation_sha,
        'train_token_count': len(tokens), 'ancestry': ancestry,
        'teacher': teacher_info,
        'selection_split': 'validation', 'test_used': False,
    }
    (args.run_dir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    events = open(args.run_dir / 'events.jsonl', 'a', encoding='utf8', buffering=1)
    best_bpb, best_step, best_kind, stale = float('inf'), None, None, 0
    history, validation_history = [], []
    train_seconds, validation_seconds = 0., 0.
    offset = torch.arange(257, device=device)
    print(json.dumps({'started': metadata}), flush=True)

    def checkpoint(predictor, step, kind, validation):
        return {
            'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
            'model': cpu_state(predictor), 'seed': args.seed,
            'train_tokens': inherited_targets + (step-start_step)*args.batch_size*256,
            'step': step, 'weight_kind': kind, 'ancestry': ancestry,
            'source_sha256': source_hashes, 'validation': validation,
            'teacher': teacher_info,
            'total_training_targets': inherited_targets + (step-start_step)*args.batch_size*256 +
                (teacher_info['total_dependency_train_tokens'] if teacher_info else 0),
        }

    model.train()
    interval_started = time.perf_counter()
    running_loss = 0.
    running_count = 0
    for step_index in range(start_step, args.steps):
        step = step_index + 1
        phase = min(1., max(0., (step_index-args.warmup) / max(1, args.steps-args.warmup)))
        lr = args.lr * min(1., step / max(1, args.warmup)) * (
            args.min_lr_ratio + (1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*phase)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        starts = torch.randint(len(tokens)-256, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + offset]
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:, :-1])
            hard_loss = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten())
            loss = hard_loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(batch[:, :-1]).float()
                temperature = args.distill_temperature
                # Sum over vocabulary, then average over tokens (not batchmean).
                soft_loss = token_distillation_kl(logits, teacher_logits, temperature)
                loss = (1-args.distill_weight)*hard_loss + args.distill_weight*soft_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at step {step}')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if ema is not None:
            decay = args.ema_decay if step > args.ema_start else 0.
            with torch.no_grad():
                for target, current in zip(ema.parameters(), model.parameters()):
                    target.lerp_(current, 1-decay)
                for target, current in zip(ema.buffers(), model.buffers()):
                    target.copy_(current)
        running_loss += loss.detach().item()
        running_count += 1
        if step % args.log_every == 0 or step == args.steps:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - interval_started
            train_seconds += elapsed
            row = {'step': step, 'loss': running_loss / running_count,
                   'last_loss': loss.item(), 'lr': lr, 'grad_norm': float(grad_norm),
                   'hard_label_loss': hard_loss.item(),
                   'train_seconds': train_seconds,
                   'processed_targets': inherited_targets+(step-start_step)*args.batch_size*256}
            history.append(row)
            events.write(json.dumps({'train': row})+'\n')
            print(json.dumps({'train': row}), flush=True)
            running_loss = 0.
            running_count = 0
            interval_started = time.perf_counter()
        if step % args.eval_every == 0 or step == args.steps or step == 1200:
            train_seconds += time.perf_counter() - interval_started
            improved = False
            for kind, predictor in [('raw', model), ('ema', ema)]:
                if predictor is None or (kind == 'ema' and step <= args.ema_start):
                    continue
                validation = score(predictor, *data['validation'], device, 'fp32')
                validation.pop('window_nll_nats')
                validation_seconds += validation['seconds']
                row = {'step': step, 'weight_kind': kind, **validation}
                validation_history.append(row)
                events.write(json.dumps({'validation': row})+'\n')
                print(json.dumps({'validation': row}), flush=True)
                if validation['bpb'] < best_bpb:
                    best_bpb, best_step, best_kind = validation['bpb'], step, kind
                    atomic_save(checkpoint(predictor, step, kind, validation), args.run_dir / 'checkpoint.pt')
                    improved = True
                if kind == 'raw':
                    atomic_save(checkpoint(predictor, step, kind, validation),
                                args.run_dir / 'snapshots' / f'step{step:06d}.pt')
                # The matched-target snapshot is retained regardless of later selection.
                if step == 1200 and start_step == 0:
                    atomic_save(checkpoint(predictor, step, kind, validation), args.run_dir / f'matched1200_{kind}.pt')
            stale = 0 if improved else stale + 1
            interval_started = time.perf_counter()
        if step % args.save_every == 0 or step == args.steps:
            state = checkpoint(model, step, 'raw', None)
            state.update(optimizer=optimizer.state_dict(), ema=cpu_state(ema) if ema is not None else None,
                         sampling_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            atomic_save(state, args.run_dir / 'resume.pt')
        summary = {**metadata, 'steps_completed': step,
                   'train_tokens': inherited_targets + (step-start_step)*args.batch_size*256,
                   'new_train_tokens': (step-start_step)*args.batch_size*256,
                   'teacher_forward_targets': (step-start_step)*args.batch_size*256 if teacher is not None else 0,
                   'teacher_member_forward_targets': (step-start_step)*args.batch_size*256*
                       teacher_info['member_forward_count'] if teacher_info else 0,
                   'total_training_targets': inherited_targets+(step-start_step)*args.batch_size*256 +
                       (teacher_info['total_dependency_train_tokens'] if teacher_info else 0),
                   'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                   'process_seconds': time.perf_counter()-process_started,
                   'best_validation_bpb': best_bpb, 'best_step': best_step, 'best_weight_kind': best_kind,
                   'history': history, 'validation_history': validation_history}
        if step % args.eval_every == 0 or step == args.steps:
            (args.run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
        if args.patience > 0 and stale >= args.patience:
            print(json.dumps({'early_stop': step, 'stale_events': stale}), flush=True)
            break
    events.close()
    summary['checkpoint_sha256'] = sha(args.run_dir / 'checkpoint.pt')
    summary['process_seconds'] = time.perf_counter()-process_started
    summary['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    (args.run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('history','validation_history','source_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
