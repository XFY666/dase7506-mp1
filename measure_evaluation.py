"""Measure peak process RAM while running the unchanged full FP32 evaluator.

The official JSON's seconds is scoring time. This wrapper also records total
process wall time and peak resident RAM, including Python, loading and scoring.
It never changes the scorer or its inputs.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil

from evidence import PROTOCOL, ROOT, file_hashes, sha, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Thread count must be positive.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_output = args.output.with_suffix('.score.json')
    stdout_path = args.output.with_suffix('.stdout.txt')
    if any(path.exists() for path in [args.output, score_output, stdout_path,
                                     score_output.with_suffix('.window-nll.npy')]):
        parser.error('Choose new output paths; existing evidence is preserved.')
    initial = snapshot(args.checkpoint)
    measurement_sources = file_hashes(['measure_evaluation.py', 'evidence.py'])
    command = [sys.executable, str(Path(__file__).with_name('evaluate.py')),
               '--checkpoint', str(args.checkpoint.resolve()), '--split', args.split,
               '--device', 'cpu', '--precision', 'fp32', '--threads', str(args.threads),
               '--output', str(score_output.resolve())]
    peak_rss, peak_commit = 0, 0
    peak_concurrent_rss, peak_concurrent_commit = 0, 0
    process_peaks = {}
    started = time.perf_counter()
    with stdout_path.open('w', encoding='utf8') as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                # Windows venv python.exe can be a small launcher whose child
                # hosts the real interpreter. Include the complete process tree.
                members = [process, *process.children(recursive=True)]
                live_rss, live_commit = 0, 0
                for member in members:
                    try:
                        info = member.memory_info()
                        rss = max(info.rss, getattr(info, 'peak_wset', 0))
                        commit = max(getattr(info, 'peak_pagefile', 0), getattr(info, 'private', 0))
                        record = process_peaks.setdefault(member.pid, dict(pid=member.pid, name=member.name(),
                                                                         peak_rss_bytes=0, peak_private_commit_bytes=0))
                        record['peak_rss_bytes'] = max(record['peak_rss_bytes'], rss)
                        record['peak_private_commit_bytes'] = max(record['peak_private_commit_bytes'], commit)
                        live_rss += info.rss
                        live_commit += getattr(info, 'private', 0)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                # Summing each observed process's lifetime peak is conservative
                # when peaks occur at different times; it never hides a child.
                peak_concurrent_rss = max(peak_concurrent_rss, live_rss)
                peak_concurrent_commit = max(peak_concurrent_commit, live_commit)
                peak_rss = max(peak_rss, live_rss, sum(p['peak_rss_bytes'] for p in process_peaks.values()))
                peak_commit = max(peak_commit, live_commit,
                                  sum(p['peak_private_commit_bytes'] for p in process_peaks.values()))
            except psutil.NoSuchProcess:
                break
            time.sleep(.02)
        return_code = child.wait()
    elapsed = time.perf_counter()-started
    final = snapshot(args.checkpoint)
    unchanged = initial == final and measurement_sources == file_hashes(measurement_sources)
    result = {'protocol': PROTOCOL, 'command': command, 'return_code': return_code,
              'wall_seconds': elapsed, 'peak_rss_bytes': peak_rss,
              'peak_private_commit_bytes': peak_commit, 'ram_limit_bytes': 4*1024**3,
              'peak_concurrent_rss_bytes': peak_concurrent_rss,
              'peak_concurrent_private_commit_bytes': peak_concurrent_commit,
              'sum_observed_process_rss_peaks_bytes': sum(p['peak_rss_bytes'] for p in process_peaks.values()),
              'sum_observed_process_commit_peaks_bytes': sum(p['peak_private_commit_bytes'] for p in process_peaks.values()),
              'ram_within_limit': peak_rss <= 4*1024**3,
              'memory_measurement_valid': bool(process_peaks),
              'platform': platform.platform(), 'python': sys.version,
              'machine': platform.machine(), 'processor': platform.processor(),
              'python_executable': sys.executable, 'torch_version': version('torch'),
              'threads': args.threads, 'source_stability_verified': unchanged,
              **initial, 'end_snapshot': final,
              'measurement_source_sha256': measurement_sources,
              'psutil_version': psutil.__version__, 'sample_interval_seconds': .02,
              'memory_method': 'Entire evaluation process tree; per-process OS lifetime peaks when available, plus sampled concurrent RSS',
              'memory_polling_limitations': '20 ms snapshots may miss short-lived descendants. OS lifetime peaks are retained for observed processes when available. The sum of individual peaks may exceed the simultaneous peak.',
              'process_peaks': list(process_peaks.values()),
              'score_output': str(score_output.resolve())}
    if return_code == 0:
        result['score'] = json.loads(score_output.read_text())
        if result['score']['checkpoint_sha256'] != initial['checkpoint_sha256']:
            unchanged = result['source_stability_verified'] = False
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if return_code:
        print(stdout_path.read_text(encoding='utf8'), file=sys.stderr)
        raise SystemExit(return_code)
    if not unchanged:
        raise RuntimeError('Checkpoint, source or benchmark bytes changed during evaluation.')


if __name__ == '__main__':
    main()
