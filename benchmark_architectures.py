"""Compare CPU cost on validation before committing to a training architecture."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from common import ROOT, autocast, make_model, setup, windows
from evaluate import score
from train_experiment import development_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-steps', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    device, precision = setup('cpu', 'fp32', args.threads)
    data = development_data()
    x, y = next(windows(data['validation'][0]))
    records = []
    for config_path in args.configs:
        config = json.loads(config_path.read_text())
        implementation = 'model' if config_path.stem == 'baseline' else 'student'
        torch.manual_seed(17)
        model, _ = make_model(implementation, config, device)
        model.eval()
        with torch.no_grad():
            for _ in range(2):
                model.predict_log_probs(x)
        row = {'config_path': str(config_path), 'config': config, 'implementation': implementation,
               'parameters': sum(p.numel() for p in model.parameters()),
               'threads': args.threads, 'split': 'validation', 'precision': 'fp32', 'cpu_seconds': []}
        for _ in range(args.repeats):
            result = score(model, *data['validation'], device, precision)
            row['cpu_seconds'].append(result['seconds'])
        if args.gpu_steps:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = model.to('cuda').train()
            batch = data['train'][0][:32*257].view(32,257).to('cuda')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, fused=True)
            timings = []
            for step in range(args.gpu_steps + 3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with autocast(torch.device('cuda'), 'bf16'):
                    loss = F.cross_entropy(model(batch[:,:-1]).float().flatten(0,1), batch[:,1:].flatten())
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if step >= 3:
                    timings.append(time.perf_counter()-started)
            row['gpu_training_seconds_per_step'] = sum(timings)/len(timings)
            row['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            row['discarded_gpu_probe_steps'] = args.gpu_steps + 3
            row['discarded_gpu_probe_train_targets'] = (args.gpu_steps + 3)*32*256
            del optimizer, batch, loss
        records.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        del model
        gc.collect()
        if args.gpu_steps:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
