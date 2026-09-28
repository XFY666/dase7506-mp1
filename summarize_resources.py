"""Validate repeated paired CPU measurements and summarize their median cost.

Run measure_evaluation.py at least three times per checkpoint in the same quiet
environment after warm-up. Use validation before freezing; the same utility can
summarize fixed-predictor test timings after freezing. It evaluates no data.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from evidence import PROTOCOL, sha


ENVIRONMENT_KEYS = ('platform', 'python', 'python_executable', 'torch_version',
                    'machine', 'processor', 'threads')


def validate_records(baseline, candidate):
    if len(baseline) < 3 or len(candidate) < 3:
        raise ValueError('Supply at least three full measurements per checkpoint.')
    records = baseline+candidate
    first = records[0]
    split = first['score'].get('split')
    coverage = {'validation': (376599, 1148007), 'test': (428405, 1292013)}
    if split not in coverage:
        raise ValueError('Resource measurements must use validation or test.')
    for row in records:
        if row.get('return_code') != 0 or not row.get('source_stability_verified'):
            raise ValueError('Every evaluation must succeed with stable dependencies.')
        if not row.get('memory_measurement_valid') or not row.get('process_peaks'):
            raise ValueError('Use valid process-tree RAM measurements.')
        if row.get('protocol') != PROTOCOL or row['score'].get('protocol') != PROTOCOL:
            raise ValueError('Resource evidence belongs to a different protocol.')
        score = row['score']
        if score.get('split') != split or score.get('precision') != 'fp32' or score.get('device') != 'cpu':
            raise ValueError('Resource evidence must use one consistent split, CPU and FP32.')
        if score['checkpoint_sha256'] != row['checkpoint_sha256']:
            raise ValueError('Resource checkpoint hashes disagree.')
        for key in ENVIRONMENT_KEYS:
            if row.get(key) != first.get(key) or key not in row:
                raise ValueError(f'Mismatched measurement environment: {key}')
        for key in ('source_sha256', 'benchmark_sha256', 'measurement_source_sha256'):
            if row.get(key) != first.get(key) or not row.get(key):
                raise ValueError(f'Mismatched fixed dependencies: {key}')
        if row.get('end_snapshot') != {name: row[name] for name in
                                      ('checkpoint_sha256', 'source_sha256', 'benchmark_sha256')}:
            raise ValueError('Start/end dependency snapshots disagree.')
        if score['seconds'] <= 0 or row['peak_rss_bytes'] <= 0:
            raise ValueError('Invalid time or memory measurement.')
        if (score['targets'], score['utf8_bytes']) != coverage[split]:
            raise ValueError('Resource evidence must cover every target and byte of the fixed split.')
        for key in ('targets', 'utf8_bytes'):
            if score[key] != first['score'][key]:
                raise ValueError('Measurements do not score the same full validation set.')
    for group in (baseline, candidate):
        if len({row['checkpoint_sha256'] for row in group}) != 1:
            raise ValueError('Each measurement group must use one unchanged checkpoint.')
        if max(row['score']['bpb'] for row in group)-min(row['score']['bpb'] for row in group) > 2e-6:
            raise ValueError('Repeated scores disagree; investigate before freezing.')
    if baseline[0]['checkpoint_sha256'] == candidate[0]['checkpoint_sha256']:
        raise ValueError('Baseline and candidate must be distinct checkpoints.')


def summarize(baseline, candidate):
    validate_records(baseline, candidate)
    base_times = [row['score']['seconds'] for row in baseline]
    candidate_times = [row['score']['seconds'] for row in candidate]
    base_median, candidate_median = statistics.median(base_times), statistics.median(candidate_times)
    peak = max(row['peak_rss_bytes'] for row in candidate)
    ratio = candidate_median/base_median
    split = candidate[0]['score']['split']
    return dict(schema='7506-resource-summary-v1', protocol=PROTOCOL, split=split,
                test_evaluated=split == 'test', precision='fp32', device='cpu',
                baseline_checkpoint_sha256=baseline[0]['checkpoint_sha256'],
                candidate_checkpoint_sha256=candidate[0]['checkpoint_sha256'],
                environment={key: candidate[0][key] for key in ENVIRONMENT_KEYS},
                source_sha256=candidate[0]['source_sha256'],
                benchmark_sha256=candidate[0]['benchmark_sha256'],
                measurement_source_sha256=candidate[0]['measurement_source_sha256'],
                baseline_seconds=base_times, candidate_seconds=candidate_times,
                baseline_median_seconds=base_median, candidate_median_seconds=candidate_median,
                median_time_ratio=ratio, time_ratio_limit=5., time_within_limit=ratio <= 5.,
                candidate_bpb=[row['score']['bpb'] for row in candidate],
                candidate_peak_rss_bytes=peak, ram_limit_bytes=4*1024**3,
                ram_within_limit=peak <= 4*1024**3,
                candidate_peak_concurrent_rss_bytes=max(row['peak_concurrent_rss_bytes'] for row in candidate),
                source_stability_verified=True, memory_measurement_valid=True,
                baseline_runs=baseline, candidate_runs=candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-records', nargs='+', type=Path, required=True)
    parser.add_argument('--candidate-records', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file; existing summaries are preserved.')
    paths = args.baseline_records+args.candidate_records
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error('Each measurement must be a distinct record file.')
    records = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    result = summarize(records[:len(args.baseline_records)], records[len(args.baseline_records):])
    result['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['record_files'] = [dict(path=str(path.resolve()), sha256=sha(path)) for path in paths]
    result['summarizer_sha256'] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('baseline_runs', 'candidate_runs')}, indent=2))


if __name__ == '__main__':
    main()
