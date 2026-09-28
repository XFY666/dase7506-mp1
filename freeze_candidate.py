"""Create a reviewable predictor freeze before the first test evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

import torch

from common import PROTOCOL, ROOT, sha
from evidence import INFERENCE_SOURCES, benchmark_hashes, file_hashes
from summarize_resources import summarize


def asset_size(path):
    size = path.stat().st_size
    expanded = size
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            expanded = sum(item.file_size for item in archive.infolist())
    return dict(path=path.relative_to(ROOT).as_posix(), file_bytes=size,
                archive_member_bytes=expanded, counted_uncompressed_bytes=max(size, expanded),
                sha256=sha(path))


def validate_evidence(checkpoint, parent_sha, sources, benchmark, validation, resource, cpu_chunk):
    """Require actual evidence for this exact predictor before any test score."""
    if int(checkpoint['config'].get('cpu_batch_chunk', 0)) != cpu_chunk:
        raise ValueError('Set cpu_batch_chunk before resource evaluation; freezing cannot change scheduling.')
    if (validation.get('protocol') != PROTOCOL or validation.get('split') != 'validation'
            or validation.get('test_used') is not False or validation.get('precision') != 'fp32'):
        raise ValueError('Use FP32 validation-only ablate_hybrid evidence.')
    if validation.get('checkpoint_sha256') != parent_sha or validation.get('selected_config') != checkpoint['config']:
        raise ValueError('Validation evidence does not describe this exact checkpoint and configuration.')
    if validation.get('benchmark_sha256') != benchmark:
        raise ValueError('Validation benchmark hashes do not match.')
    for name, digest in sources.items():
        if validation.get('source_sha256', {}).get(name) != digest:
            raise ValueError(f'Stale validation implementation: {name}')
    if resource.get('schema') != '7506-resource-summary-v1':
        raise ValueError('Use the repeated-run JSON produced by summarize_resources.py.')
    if resource.get('split') != 'validation' or resource.get('test_evaluated') is not False:
        raise ValueError('A pre-test freeze requires validation-only resource evidence.')
    recomputed = summarize(resource['baseline_runs'], resource['candidate_runs'])
    for name, expected in recomputed.items():
        if name not in ('baseline_runs', 'candidate_runs') and resource.get(name) != expected:
            raise ValueError(f'Resource summary disagrees with its underlying measurements: {name}')
    if recomputed['candidate_checkpoint_sha256'] != parent_sha:
        raise ValueError('Resource measurements must use the exact selected checkpoint.')
    if recomputed['source_sha256'] != sources or recomputed['benchmark_sha256'] != benchmark:
        raise ValueError('Resource evidence uses stale source or benchmark bytes.')
    if recomputed['measurement_source_sha256'] != file_hashes(['measure_evaluation.py', 'evidence.py']):
        raise ValueError('Resource evidence uses a different measurement wrapper or dependency helper.')
    if not recomputed['time_within_limit'] or not recomputed['ram_within_limit']:
        raise ValueError('Candidate does not pass the measured CPU time and RAM limits.')
    full = [row for row in validation['rows'] if row['name'] == 'full']
    if len(full) != 1 or any(abs(score-full[0]['bpb']) > 2e-6 for score in recomputed['candidate_bpb']):
        raise ValueError('Validation and resource runs disagree on the full predictor score.')
    return recomputed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu-batch-chunk', type=int, default=8)
    parser.add_argument('--validation-evidence', type=Path, required=True)
    parser.add_argument('--resource-evidence', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        parser.error('Place the frozen predictor inside the submitted code directory.')
    if destination.exists() and any(destination.iterdir()):
        parser.error('Choose a new empty directory; existing freezes are immutable.')
    if args.cpu_batch_chunk < 0:
        parser.error('CPU batch chunk must be nonnegative.')
    for path in [args.checkpoint, args.validation_evidence, args.resource_evidence]:
        if not path.is_file():
            parser.error(f'Missing evidence: {path}')
    parent_sha = sha(args.checkpoint)
    evidence_inputs = {str(path.resolve()): sha(path) for path in
                       [args.validation_evidence, args.resource_evidence]}
    sources = file_hashes(INFERENCE_SOURCES)
    benchmark = benchmark_hashes()
    freeze_sources = file_hashes(['freeze_candidate.py', 'verify_frozen.py', 'evidence.py',
                                  'summarize_resources.py', 'measure_evaluation.py'])
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] != 'hybrid':
        raise ValueError('Expected a selected hybrid checkpoint.')
    checkpoint = dict(checkpoint)
    validation = json.loads(args.validation_evidence.read_text(encoding='utf-8'))
    resource = json.loads(args.resource_evidence.read_text(encoding='utf-8'))
    checked_resource = validate_evidence(checkpoint, parent_sha, sources, benchmark,
                                         validation, resource, args.cpu_batch_chunk)
    asset = (ROOT/checkpoint['config']['ngram_asset']).resolve()
    if not asset.is_relative_to(ROOT.resolve()) or sha(asset) != checkpoint['config']['ngram_sha256']:
        raise ValueError('Count asset does not match the selected predictor.')
    frozen_at = datetime.now(timezone.utc).isoformat()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/'checkpoint.pt'
    # Keep exactly the bytes used by validation and resource measurements.
    # Freeze metadata belongs in the adjacent manifest, not the checkpoint.
    shutil.copyfile(args.checkpoint, target)
    if sha(target) != parent_sha:
        raise RuntimeError('Frozen checkpoint copy does not match its measured parent.')
    assets = [asset_size(target), asset_size(asset), *[asset_size(ROOT/name) for name in INFERENCE_SOURCES]]
    total = sum(item['counted_uncompressed_bytes'] for item in assets)
    if total > 64*1024**2:
        raise ValueError(f'Inference assets exceed 64 MiB: {total}')
    evidence_copies = []
    for name, original in [('validation_evidence.json', args.validation_evidence),
                           ('resource_evidence.json', args.resource_evidence)]:
        copied = destination/name
        copied.write_bytes(original.read_bytes())
        evidence_copies.append(dict(path=copied.relative_to(ROOT).as_posix(), sha256=sha(copied)))
    if (sha(args.checkpoint) != parent_sha or file_hashes(INFERENCE_SOURCES) != sources
            or benchmark_hashes() != benchmark or file_hashes(freeze_sources) != freeze_sources
            or sha(asset) != checkpoint['config']['ngram_sha256']
            or any(sha(Path(path)) != digest for path, digest in evidence_inputs.items())):
        raise RuntimeError('Checkpoint, evidence or dependencies changed while freezing.')
    manifest = dict(protocol=PROTOCOL, frozen_at_utc=frozen_at,
                    selection_split='validation', test_evaluated_at_freeze=False,
                    parent_checkpoint_sha256=parent_sha,
                    implementation=checkpoint['implementation'], config=checkpoint['config'],
                    inference_source_sha256=sources, benchmark_sha256=benchmark,
                    freeze_tool_sha256=freeze_sources, evidence=evidence_copies, assets=assets,
                    inference_assets_uncompressed_bytes=total,
                    inference_asset_limit_bytes=64*1024**2,
                    validation_evidence_sha256=sha(args.validation_evidence),
                    resource_evidence_sha256=sha(args.resource_evidence),
                    cpu_median_time_ratio=checked_resource['median_time_ratio'],
                    peak_rss_bytes=checked_resource['candidate_peak_rss_bytes'],
                    parameter_weights_changed=False,
                    checkpoint_bytes_changed=False,
                    scheduling_changed=False, cpu_batch_chunk=args.cpu_batch_chunk,
                    rule='No predictor, asset or setting changes after reading test results.')
    (destination/'freeze.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
