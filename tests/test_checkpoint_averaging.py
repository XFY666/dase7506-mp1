"""Checkpoint averaging must preserve ties, identity and ancestry cost."""
import json
import tempfile
from pathlib import Path
import unittest

import torch

from average_checkpoints import average_states, candidate_groups, load_parents, make_average
from common import PROTOCOL


class CheckpointAveragingTests(unittest.TestCase):
    def test_mean_uses_fp32_and_retains_tied_storage(self):
        reference_weight = torch.zeros(2, 2)
        reference = {'token.weight': reference_weight, 'head.weight': reference_weight,
                     'counter': torch.tensor(5)}
        states = []
        for value in (1., 5.):
            weight = torch.full((2, 2), value, dtype=torch.float16)
            states.append({'token.weight': weight, 'head.weight': weight, 'counter': torch.tensor(5)})
        result = average_states(states, reference)
        torch.testing.assert_close(result['token.weight'], torch.full((2, 2), 3.))
        self.assertEqual(result['token.weight'].dtype, torch.float32)
        self.assertEqual(result['token.weight'].data_ptr(), result['head.weight'].data_ptr())
        self.assertEqual(result['counter'].item(), 5)

    def test_inconsistent_tied_tensors_are_rejected(self):
        shared = torch.zeros(2, 2)
        reference = {'token.weight': shared, 'head.weight': shared}
        with self.assertRaisesRegex(ValueError, 'Inconsistent tied'):
            average_states([{'token.weight': torch.ones(2, 2), 'head.weight': torch.zeros(2, 2)}], reference)

    def test_parent_training_cost_is_maximum_and_inference_scalars_reset(self):
        parents = []
        for step, value in ((2, 1.), (4, 3.)):
            parents.append(dict(checkpoint={
                'config': {'width': 2, 'cache_weight': .2, 'cache_temperature': 15., 'logit_temperature': .95},
                'model': {'weight': torch.tensor([value])}, 'step': step, 'train_tokens': step * 100,
                'ancestry': [{'sha256': 'shared-parent', 'train_tokens': 100}],
                'weight_kind': 'raw', 'optimizer': {},
            }, info={'step': step, 'train_tokens': step * 100, 'sha256': str(step)}, run_directory=Path('run')))
        result = make_average(parents, {'weight': torch.zeros(1)}, 'test-script-hash')
        torch.testing.assert_close(result['model']['weight'], torch.tensor([2.]))
        self.assertEqual(result['train_tokens'], 400)
        self.assertEqual(result['averaging']['additional_train_targets'], 0)
        self.assertEqual(result['config']['cache_weight'], 0.)
        self.assertEqual(result['config']['logit_temperature'], 1.)
        self.assertNotIn('optimizer', result)
        self.assertEqual(len(result['averaging']['parent_checkpoints']), 2)

    def test_candidate_groups_are_distinct_and_cover_best_and_tail(self):
        parents = [dict(info={'step': i + 1, 'recorded_validation_bpb': abs(i - 4) + 1.}) for i in range(10)]
        groups = candidate_groups(parents, [2, 4, 8])
        self.assertEqual(groups[0], ('best_raw', [4]))
        self.assertIn(('before_best_4', [1, 2, 3, 4]), groups)
        self.assertIn(('tail_8', list(range(2, 10))), groups)
        self.assertEqual(len(groups), len({tuple(indices) for _, indices in groups}))

    def test_parents_must_come_from_one_recorded_run(self):
        # Keep temporary files next to the test so that relative paths resolve on any drive.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            paths = []
            config = {'width': 2, 'heads': 1, 'depth': 1, 'context': 256, 'vocab': 2048}
            for run_name in ('run_a', 'run_b'):
                run = root / run_name
                snapshots = run / 'snapshots'
                snapshots.mkdir(parents=True)
                (run / 'run.json').write_text(json.dumps(dict(seed=17, implementation='student', config=config)))
                path = snapshots / 'step000001.pt'
                torch.save(dict(protocol=PROTOCOL, seed=17, implementation='student', config=config,
                                model={'weight': torch.ones(2)}, step=1, train_tokens=100,
                                weight_kind='raw', ancestry=[], source_sha256={'student.py': 'same'},
                                validation={'bpb': 2.}), path)
                paths.append(path)
            loaded = load_parents([paths[0]])
            self.assertEqual(loaded[0]['info']['step'], 1)
            self.assertEqual(len(loaded[0]['info']['sha256']), 64)
            with self.assertRaisesRegex(ValueError, 'same recorded run'):
                load_parents(paths)


if __name__ == '__main__':
    unittest.main()
