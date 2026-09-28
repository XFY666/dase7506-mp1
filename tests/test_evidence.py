"""Stale or incomparable evidence must not authorize a pre-test freeze."""
import copy
import unittest

from evidence import PROTOCOL, checked_relative_path, file_hashes
from freeze_candidate import validate_evidence
from summarize_resources import summarize


def run_record(checkpoint, seconds, bpb):
    sources, benchmark = {'student.py': 'source'}, {'data/manifest.json': 'benchmark'}
    record = dict(protocol=PROTOCOL, return_code=0, checkpoint_sha256=checkpoint,
                  source_stability_verified=True, memory_measurement_valid=True,
                  process_peaks=[dict(pid=123, peak_rss_bytes=2000)],
                  peak_rss_bytes=2000, peak_concurrent_rss_bytes=1900,
                  platform='same', python='3.12', python_executable='python',
                  torch_version='2.7.1+cpu', machine='AMD64', processor='same', threads=4,
                  source_sha256=sources, benchmark_sha256=benchmark,
                  measurement_source_sha256=file_hashes(['measure_evaluation.py', 'evidence.py']),
                  score=dict(protocol=PROTOCOL, split='validation', precision='fp32', device='cpu',
                             checkpoint_sha256=checkpoint, seconds=seconds, bpb=bpb,
                             targets=376599, utf8_bytes=1148007))
    record['end_snapshot'] = {name: record[name] for name in
                              ('checkpoint_sha256', 'source_sha256', 'benchmark_sha256')}
    return record


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = [run_record('baseline', t, 2.1) for t in [1., 1.2, 1.1]]
        self.candidate = [run_record('candidate', t, 1.5) for t in [3.3, 3.6, 3.5]]

    def test_medians_require_repeated_comparable_full_runs(self):
        summary = summarize(self.baseline, self.candidate)
        self.assertAlmostEqual(summary['median_time_ratio'], 3.5/1.1)
        self.assertEqual(summary['candidate_peak_rss_bytes'], 2000)
        self.assertTrue(summary['ram_within_limit'] and summary['time_within_limit'])
        for field, value in [('threads', 8), ('memory_measurement_valid', False),
                             ('source_stability_verified', False)]:
            changed = copy.deepcopy(self.candidate)
            changed[1][field] = value
            with self.assertRaises(ValueError):
                summarize(self.baseline, changed)
        changed = copy.deepcopy(self.candidate)
        changed[0]['score']['targets'] -= 1
        with self.assertRaises(ValueError):
            summarize(self.baseline, changed)
        with self.assertRaises(ValueError):
            summarize(self.baseline[:2], self.candidate)

    def test_exact_predictor_sources_and_scheduling_are_required(self):
        resource = summarize(self.baseline, self.candidate)
        sources, benchmark = resource['source_sha256'], resource['benchmark_sha256']
        checkpoint = {'config': {'cpu_batch_chunk': 8, 'cache_weight': .2}}
        validation = dict(protocol=PROTOCOL, split='validation', test_used=False,
                          precision='fp32', checkpoint_sha256='candidate',
                          selected_config=checkpoint['config'], source_sha256=sources,
                          benchmark_sha256=benchmark, rows=[dict(name='full', bpb=1.5)])
        validate_evidence(checkpoint, 'candidate', sources, benchmark, validation, resource, 8)
        with self.assertRaises(ValueError):
            validate_evidence(checkpoint, 'candidate', sources, benchmark, validation, resource, 4)
        for field, value in [('checkpoint_sha256', 'stale'),
                             ('selected_config', {'cpu_batch_chunk': 8, 'cache_weight': .3}),
                             ('source_sha256', {'student.py': 'old'})]:
            changed = copy.deepcopy(validation)
            changed[field] = value
            with self.assertRaises(ValueError):
                validate_evidence(checkpoint, 'candidate', sources, benchmark, changed, resource, 8)

    def test_summary_tampering_and_resource_failure_are_rejected(self):
        resource = summarize(self.baseline, self.candidate)
        sources, benchmark = resource['source_sha256'], resource['benchmark_sha256']
        checkpoint = {'config': {'cpu_batch_chunk': 8}}
        validation = dict(protocol=PROTOCOL, split='validation', test_used=False,
                          precision='fp32', checkpoint_sha256='candidate',
                          selected_config=checkpoint['config'], source_sha256=sources,
                          benchmark_sha256=benchmark, rows=[dict(name='full', bpb=1.5)])
        changed = copy.deepcopy(resource)
        changed['median_time_ratio'] = 1.
        with self.assertRaises(ValueError):
            validate_evidence(checkpoint, 'candidate', sources, benchmark, validation, changed, 8)
        slow = [run_record('candidate', t, 1.5) for t in [8., 8., 8.]]
        with self.assertRaises(ValueError):
            validate_evidence(checkpoint, 'candidate', sources, benchmark, validation,
                              summarize(self.baseline, slow), 8)

    def test_dependency_paths_cannot_escape(self):
        with self.assertRaises(ValueError):
            checked_relative_path('../outside.pt')

    def test_freeze_rejects_measurement_wrapper_change(self):
        baseline, candidate = copy.deepcopy(self.baseline), copy.deepcopy(self.candidate)
        for row in baseline+candidate:
            row['measurement_source_sha256']['measure_evaluation.py'] = 'old-wrapper'
        resource = summarize(baseline, candidate)
        sources, benchmark = resource['source_sha256'], resource['benchmark_sha256']
        checkpoint = {'config': {'cpu_batch_chunk': 8}}
        validation = dict(protocol=PROTOCOL, split='validation', test_used=False,
                          precision='fp32', checkpoint_sha256='candidate',
                          selected_config=checkpoint['config'], source_sha256=sources,
                          benchmark_sha256=benchmark, rows=[dict(name='full', bpb=1.5)])
        with self.assertRaisesRegex(ValueError, 'different measurement wrapper'):
            validate_evidence(checkpoint, 'candidate', sources, benchmark, validation, resource, 8)

    def test_test_resource_summary_is_allowed_only_after_freeze(self):
        baseline, candidate = copy.deepcopy(self.baseline), copy.deepcopy(self.candidate)
        for row in baseline+candidate:
            row['score'].update(split='test', targets=428405, utf8_bytes=1292013)
        resource = summarize(baseline, candidate)
        self.assertEqual(resource['split'], 'test')
        self.assertTrue(resource['test_evaluated'])
        with self.assertRaises(ValueError):
            summarize(self.baseline, candidate)
        sources, benchmark = resource['source_sha256'], resource['benchmark_sha256']
        checkpoint = {'config': {'cpu_batch_chunk': 8}}
        validation = dict(protocol=PROTOCOL, split='validation', test_used=False,
                          precision='fp32', checkpoint_sha256='candidate',
                          selected_config=checkpoint['config'], source_sha256=sources,
                          benchmark_sha256=benchmark, rows=[dict(name='full', bpb=1.5)])
        with self.assertRaisesRegex(ValueError, 'validation-only resource evidence'):
            validate_evidence(checkpoint, 'candidate', sources, benchmark, validation, resource, 8)


if __name__ == '__main__':
    unittest.main()
