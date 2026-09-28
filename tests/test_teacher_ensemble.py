"""Frozen teacher probabilities, embedded aliases and unique training ancestry."""
import io
import unittest

import torch
from torch.nn import functional as F

from build_teacher_ensemble import aggregate_nll, target_log_probabilities, unique_ancestry
from common import PROTOCOL, ROOT, sha
from teacher_ensemble import TeacherEnsemble
from train_experiment import cpu_state, token_distillation_kl, validate_teacher_checkpoint


class TeacherEnsembleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        member = dict(vocab=2048, context=256, width=16, heads=2, depth=1,
                      mlp_hidden=32, dropout=.8, embedding_dropout=.8)
        self.config = dict(vocab=2048, context=256, training_only=True,
                           member_implementations=['student', 'student'],
                           member_configs=[member, dict(member, depth=2)],
                           alpha=.25, common_temperature=1.05)
        self.model = TeacherEnsemble(self.config)
        self.ids = torch.randint(0, 2048, (2, 9))

    def test_probabilities_match_explicit_mixture_and_endpoints(self):
        with torch.no_grad():
            first = F.log_softmax(self.model.members[0](self.ids)/1.05, -1)
            second = F.log_softmax(self.model.members[1](self.ids)/1.05, -1)
            for alpha in [0., .25, 1.]:
                self.model.config['alpha'] = alpha
                actual = self.model(self.ids)
                expected = (alpha*first.exp()+(1-alpha)*second.exp()).log()
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(actual.logsumexp(-1), torch.zeros_like(actual[...,0]), atol=1e-6, rtol=0.)
            self.model.config['alpha'] = 1.
            torch.testing.assert_close(self.model(self.ids), first, atol=0., rtol=0.)

    def test_teacher_stays_eval_detached_and_student_receives_gradient(self):
        self.model.train()
        self.assertFalse(self.model.training)
        self.assertTrue(all(not member.training for member in self.model.members))
        targets = self.model(self.ids)
        self.assertFalse(targets.requires_grad)
        student = torch.randn_like(targets, requires_grad=True)
        token_distillation_kl(student, targets, 1.).backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in self.model.parameters()))

    def test_causality_reset_and_embedded_tied_storage(self):
        full = self.model(self.ids)
        torch.testing.assert_close(full[:, :5], self.model(self.ids[:, :5]), atol=2e-6, rtol=1e-6)
        changed = self.ids.clone()
        changed[:, 5:] = (changed[:, 5:]+31) % 2048
        torch.testing.assert_close(full[:, :5], self.model(changed)[:, :5], atol=0., rtol=0.)
        state = cpu_state(self.model)
        for index in [0, 1]:
            self.assertEqual(state[f'members.{index}.token.weight'].data_ptr(), state[f'members.{index}.head.weight'].data_ptr())
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=True)
        self.assertEqual(restored['members.0.token.weight'].data_ptr(), restored['members.0.head.weight'].data_ptr())
        clone = TeacherEnsemble(self.config)
        clone.load_state_dict(restored, strict=True)
        torch.testing.assert_close(full, clone(self.ids), atol=0., rtol=0.)

    def test_scalar_sweep_matches_full_teacher_with_masked_targets(self):
        targets = (self.ids+1) % 2048
        targets[-1, -2:] = -100
        temperatures, alphas = [.95, 1.1], [0., .25, .5, 1.]
        first = target_log_probabilities(self.model.members[0](self.ids), targets, temperatures)
        second = target_log_probabilities(self.model.members[1](self.ids), targets, temperatures)
        aggregate = aggregate_nll(first, second, alphas, targets != -100)
        for i, temperature in enumerate(temperatures):
            for j, alpha in enumerate(alphas):
                self.model.config.update(common_temperature=temperature, alpha=alpha)
                losses = -self.model(self.ids).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                actual = losses.masked_fill(targets == -100, 0.).double().sum()
                torch.testing.assert_close(aggregate[i,j], actual, atol=2e-5, rtol=1e-7)

    def test_unique_segments_do_not_duplicate_resume_prefix_or_snapshot(self):
        def segment(name, start, new):
            return dict(segment_id=name, inherited_targets=start, new_targets=new)
        six = dict(train_tokens=163840000, segments=[segment('six', 0, 163840000)])
        eight = dict(train_tokens=163840000, segments=[segment('eight-prefix', 0, 49152000),
                     segment('eight-resume', 49152000, 114688000)])
        prefix = dict(train_tokens=49152000, segments=[segment('eight-prefix', 0, 49152000)])
        count, segments = unique_ancestry([six, eight, prefix, eight])
        self.assertEqual(count, 327680000)
        self.assertEqual(len(segments), 3)
        with self.assertRaises(ValueError):
            unique_ancestry([six, dict(train_tokens=100, segments=[segment('six', 50, 100)])])

    def test_training_harness_enforces_t1_and_teacher_source_hashes(self):
        checkpoint = dict(protocol=PROTOCOL, implementation='teacher_ensemble',
                          training_only=True, final_submission_allowed=False, kd_temperature_supported=1.,
                          inference_source_sha256={name:sha(ROOT/name) for name in ['teacher_ensemble.py','student.py']})
        validate_teacher_checkpoint(checkpoint, 1.)
        with self.assertRaises(ValueError):
            validate_teacher_checkpoint(checkpoint, 2.)
        checkpoint['inference_source_sha256']['student.py'] = 'stale'
        with self.assertRaises(ValueError):
            validate_teacher_checkpoint(checkpoint, 1.)


if __name__ == '__main__':
    unittest.main()
