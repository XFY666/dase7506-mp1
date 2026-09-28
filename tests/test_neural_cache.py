"""Independent checks of strict cache alignment and whole-model causality."""
import unittest
from unittest import mock

import torch

from student import build_model, neural_cache_distribution


class NeuralCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        self.model = build_model(dict(
            vocab=2048, context=256, width=32, heads=4, depth=2,
            cache_weight=.3, cache_temperature=15., cache_decay=.02,
            cache_min_history=2,
        )).eval()

    def test_cache_keys_have_only_already_observed_successors(self):
        ids = torch.tensor([[9, 7, 8, 7]])
        hidden = torch.eye(4)[None]
        weights, active = neural_cache_distribution(ids, hidden, temperature=0.)
        expected = torch.tensor([[[0., 0., 0.], [1., 0., 0.],
                                  [.5, .5, 0.], [1/3, 1/3, 1/3]]])
        torch.testing.assert_close(weights, expected)
        self.assertEqual(active.flatten().tolist(), [False, True, True, True])
        probs = torch.zeros(1, 4, 10)
        probs.scatter_add_(-1, ids[:, None, 1:].expand(1, 4, 3), weights)
        self.assertEqual(probs[0, 1, 7].item(), 1.)
        torch.testing.assert_close(probs[0, 2, 7:9], torch.tensor([.5, .5]))
        torch.testing.assert_close(probs[0, 3, 7:9], torch.tensor([2/3, 1/3]))

    def test_tuner_refuses_source_changes(self):
        from tune_cache import assert_sources_unchanged
        expected = {'student.py': 'model-digest', 'tune_cache.py': 'tuner-digest'}
        actual = dict(expected)
        with mock.patch('tune_cache.sha', side_effect=lambda path: actual[path.name]):
            assert_sources_unchanged(expected)
            actual['student.py'] = 'changed-model-digest'
            with self.assertRaisesRegex(RuntimeError, 'student.py'):
                assert_sources_unchanged(expected)

    def test_cache_layer_preserves_default_and_selects_raw_block_output(self):
        ids = torch.tensor([[4, 7, 4, 8, 4, 7]])
        with torch.no_grad():
            original_features = self.model.features(ids)
            original_logits = self.model(ids)
            final, cached = self.model.features_for_cache(ids)
            self.assertIs(final, cached)
            torch.testing.assert_close(final, original_features, atol=0., rtol=0.)
            first_block = self.model.blocks[0](self.model.embedding_dropout(self.model.token(ids)))
            self.model.config['cache_layer'] = 1
            final, cached = self.model.features_for_cache(ids)
            torch.testing.assert_close(final, original_features, atol=0., rtol=0.)
            torch.testing.assert_close(cached, first_block, atol=0., rtol=0.)
            torch.testing.assert_close(self.model(ids), original_logits, atol=0., rtol=0.)
            with mock.patch('student.neural_cache_distribution', wraps=neural_cache_distribution) as cache_call:
                self.model.predict_probabilities(ids)
                torch.testing.assert_close(cache_call.call_args.args[1], first_block, atol=0., rtol=0.)
            self.model.config['cache_layer'] = 0
            with self.assertRaisesRegex(ValueError, 'cache_layer'):
                self.model.features_for_cache(ids)

    def test_intermediate_cache_is_causal_normalized_and_independent(self):
        ids = torch.tensor([[4, 7, 4, 8, 4, 7, 4, 8], [6, 3, 6, 3, 6, 9, 6, 3]])
        with torch.no_grad():
            for layer in (1, 2):
                self.model.config.update(cache_layer=layer, cache_min_similarity=.5)
                probs = self.model.predict_probabilities(ids)
                logs = self.model.predict_log_probs(ids)
                torch.testing.assert_close(probs.sum(-1), torch.ones(2, 8), atol=1e-6, rtol=0.)
                torch.testing.assert_close(probs, logs.exp(), atol=3e-7, rtol=2e-6)
                for length in (1, 2, 4, 7):
                    prefix = self.model.predict_probabilities(ids[:, :length])
                    torch.testing.assert_close(probs[:, :length], prefix, atol=3e-7, rtol=2e-6)
                alone = self.model.predict_probabilities(ids[:1])
                torch.testing.assert_close(probs[:1], alone, atol=3e-7, rtol=2e-6)
                self.model.predict_probabilities((ids + 53) % 2048)
                torch.testing.assert_close(probs, self.model.predict_probabilities(ids), atol=0., rtol=0.)

    def test_intermediate_cache_tuning_matches_full_predictor(self):
        from tune_cache import (accumulate_mixture_nll, target_base_probabilities,
                                target_cache_probabilities)
        ids = torch.tensor([[4, 7, 4, 8, 4, 7]])
        targets = torch.tensor([[7, 4, 8, 4, 7, 4]])
        self.model.config.update(cache_layer=1, logit_temperature=1.1,
                                 cache_weight=.2, cache_temperature=24., cache_decay=.01,
                                 cache_min_history=2, cache_min_similarity=.5)
        with torch.no_grad():
            final, cached = self.model.features_for_cache(ids)
            base = target_base_probabilities(self.model.head(final), targets, [1.1])
            cache, active = target_cache_probabilities(ids, cached, targets, [(24., .01, 2, .5)])
            aggregate = accumulate_mixture_nll(base, cache, active, [.2], targets != -100)[0, 0, 0]
            full = -self.model.predict_log_probs(ids).gather(-1, targets[..., None])[..., 0].double().sum()
            torch.testing.assert_close(aggregate, full, atol=1e-5, rtol=1e-6)

    def test_each_prediction_equals_the_same_prefix_scored_alone(self):
        ids = torch.tensor([[4, 7, 5, 4, 7, 8, 4, 7, 5, 4, 7, 5]])
        with torch.no_grad():
            for minimum_similarity in (-1., .5, .9):
                self.model.config['cache_min_similarity'] = minimum_similarity
                full = self.model.predict_log_probs(ids)
                for length in (1, 2, 3, 6, 9, 12):
                    prefix = self.model.predict_log_probs(ids[:, :length])
                    torch.testing.assert_close(full[:, :length], prefix, atol=2e-6, rtol=1e-6)

    def test_similarity_gate_excludes_current_and_future_keys(self):
        ids = torch.tensor([[9, 7, 8, 7]])
        hidden = torch.tensor([[[1., 0.], [0., 1.], [0., 1.], [1., 0.]]])
        # At position 1 the same-position and future vectors are exact matches;
        # neither has an observed successor and neither may open the gate.
        _, active = neural_cache_distribution(ids, hidden, temperature=0., decay=100., min_similarity=.8)
        self.assertEqual(active.flatten().tolist(), [False, False, True, True])
        changed = hidden.clone()
        changed[:, 2:] *= -1.
        _, changed_active = neural_cache_distribution(ids, changed, min_similarity=.8)
        torch.testing.assert_close(active[:, :2], changed_active[:, :2])
        _, minimum = neural_cache_distribution(ids, hidden, min_history=3, min_similarity=.8)
        self.assertEqual(minimum.flatten().tolist(), [False, False, False, True])
        # Even a caller requesting zero history cannot enable the first row.
        _, first = neural_cache_distribution(ids, hidden, min_history=0, min_similarity=-1.)
        self.assertFalse(first[0, 0, 0])

    def test_future_tokens_cannot_change_previous_cache_probabilities(self):
        ids = torch.randint(0, 2048, (2, 17))
        changed = ids.clone()
        changed[:, 8:] = (changed[:, 8:] + 53) % 2048
        with torch.no_grad():
            first = self.model.predict_log_probs(ids)
            other = self.model.predict_log_probs(changed)
        torch.testing.assert_close(first[:, :8], other[:, :8], atol=1e-6, rtol=1e-6)

    def test_normalization_single_token_batch_independence_and_state_reset(self):
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            together = self.model.predict_log_probs(ids)
            alone = self.model.predict_log_probs(ids[:1])
            self.model.predict_log_probs((ids + 31) % 2048)
            again = self.model.predict_log_probs(ids)
            single = self.model.predict_log_probs(ids[:, :1])
        self.assertTrue(torch.isfinite(together).all())
        torch.testing.assert_close(together.logsumexp(-1), torch.zeros(2, 12), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(single.logsumexp(-1), torch.zeros(2, 1), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(together[:1], alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(together, again, atol=1e-6, rtol=1e-6)

    def test_tuning_aggregates_equal_predictor_losses(self):
        from tune_cache import (accumulate_mixture_nll, target_base_probabilities,
                                target_cache_probabilities)
        ids = torch.tensor([[4, 7, 4, 8, 4, 7], [6, 3, 6, 3, 6, 9]])
        targets = torch.tensor([[7, 4, 8, 4, 7, 4], [3, 6, 3, 6, -100, -100]])
        temperatures = [.9, 1.1]
        settings = [(0., 0., 1), (15., .02, 3), (15., .02, 1, .8), (15., .02, 3, .99)]
        strengths = [0., .3]
        with torch.no_grad():
            hidden = self.model.features(ids)
            logits = self.model.head(hidden).float()
            base = target_base_probabilities(logits, targets, temperatures)
            cache, active = target_cache_probabilities(ids, hidden, targets, settings)
            aggregates = accumulate_mixture_nll(base, cache, active, strengths, targets != -100)
            for ti, temperature in enumerate(temperatures):
                for ci, setting in enumerate(settings):
                    cache_temperature, decay, minimum = setting[:3]
                    minimum_similarity = setting[3] if len(setting) == 4 else -1.
                    for wi, strength in enumerate(strengths):
                        self.model.config.update(logit_temperature=temperature,
                                                 cache_temperature=cache_temperature,
                                                 cache_decay=decay, cache_min_history=minimum,
                                                 cache_min_similarity=minimum_similarity,
                                                 cache_weight=strength)
                        logp = self.model.predict_log_probs(ids)
                        losses = -logp.gather(-1, targets.clamp_min(0)[..., None])[..., 0]
                        expected = losses.masked_fill(targets == -100, 0.).double().sum()
                        torch.testing.assert_close(aggregates[ti, ci, wi], expected, atol=1e-5, rtol=1e-6)

    def test_scatter_mixture_agrees_with_previous_logspace_formula(self):
        ids = torch.tensor([[4, 7, 4, 8, 4, 7], [6, 3, 6, 3, 6, 9]])
        hidden = torch.randn(2, 6, 32)
        # Very unlikely neural tokens remain representable above the FP32 floor.
        logits = torch.linspace(-70., 0., 2048)[None, None].expand(2, 6, -1)
        neural = logits.log_softmax(-1)
        weights, active = neural_cache_distribution(ids, hidden, temperature=15., decay=.02, min_history=2)
        values = ids[:, None, 1:].expand(-1, ids.shape[1], -1)
        cache = torch.zeros_like(neural).scatter_add_(-1, values, weights)
        tiny = torch.finfo(torch.float32).tiny
        with torch.no_grad(), mock.patch.object(self.model, 'features', return_value=hidden), \
                mock.patch.object(self.model.head, 'forward', return_value=logits):
            for strength in (.3, 1e-8, 1e-30):
                self.model.config['cache_weight'] = strength
                actual = self.model.predict_log_probs(ids)
                coefficient = active.float() * strength
                reference = torch.logaddexp(neural + torch.log1p(-coefficient),
                                           cache.clamp_min(tiny).log() + coefficient.log())
                torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-6)
                torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2, 6), atol=1e-6, rtol=0.)

    def test_extreme_tail_floor_is_finite_and_preserves_probability_mass(self):
        ids = torch.tensor([[4, 7, 4, 8, 4, 7]])
        hidden = torch.randn(1, 6, 32)
        logits = torch.full((1, 6, 2048), -1000.)
        logits[..., 0] = 0.
        logits[..., 1] = -80.
        neural = logits.log_softmax(-1)
        weights, active = neural_cache_distribution(ids, hidden, temperature=48., decay=.02, min_history=2)
        values = ids[:, None, 1:].expand(-1, ids.shape[1], -1)
        cache = torch.zeros_like(neural).scatter_add_(-1, values, weights)
        tiny = torch.finfo(torch.float32).tiny
        self.model.config['cache_temperature'] = 48.
        with torch.no_grad(), mock.patch.object(self.model, 'features', return_value=hidden), \
                mock.patch.object(self.model.head, 'forward', return_value=logits):
            for strength in (.3, 1e-35):
                self.model.config['cache_weight'] = strength
                actual = self.model.predict_log_probs(ids)
                coefficient = active.float() * strength
                reference = torch.logaddexp(neural + torch.log1p(-coefficient),
                                           cache.clamp_min(tiny).log() + coefficient.log())
                self.assertTrue(torch.isfinite(actual).all())
                # The old formula floors cache mass; the optimized formula floors
                # final mass. Their extreme-tail logs differ, but probability
                # differences remain at the FP32 floor, below useful mass precision.
                torch.testing.assert_close(actual.exp(), reference.exp(), atol=2*tiny, rtol=2e-5)
                torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(1, 6), atol=1e-6, rtol=0.)

    def test_probability_api_matches_logs_and_is_causal_and_independent(self):
        ids = torch.tensor([[4, 7, 4, 8, 4, 7, 4, 8], [6, 3, 6, 3, 6, 9, 6, 3]])
        with torch.no_grad():
            for strength, threshold in ((0., -1.), (.3, -1.), (.3, .8)):
                self.model.config.update(cache_weight=strength, cache_min_similarity=threshold,
                                         logit_temperature=.95)
                probs = self.model.predict_probabilities(ids)
                log_probs = self.model.predict_log_probs(ids)
                self.assertEqual(probs.dtype, torch.float32)
                self.assertTrue(torch.isfinite(probs).all())
                self.assertTrue((probs >= 0.).all())
                torch.testing.assert_close(probs.sum(-1), torch.ones(2, 8), atol=1e-6, rtol=0.)
                torch.testing.assert_close(probs, log_probs.exp(), atol=3e-7, rtol=2e-6)
                for length in (1, 2, 4, 7):
                    prefix = self.model.predict_probabilities(ids[:, :length])
                    torch.testing.assert_close(probs[:, :length], prefix, atol=3e-7, rtol=2e-6)
                alone = self.model.predict_probabilities(ids[:1])
                torch.testing.assert_close(probs[:1], alone, atol=3e-7, rtol=2e-6)
                self.model.predict_probabilities((ids + 53) % 2048)
                torch.testing.assert_close(probs, self.model.predict_probabilities(ids), atol=0., rtol=0.)

    def test_probability_api_extreme_tails_match_log_probability_mass(self):
        ids = torch.tensor([[4, 7, 4, 8, 4, 7]])
        hidden = torch.randn(1, 6, 32)
        logits = torch.full((1, 6, 2048), -1000.)
        logits[..., 0] = 0.
        logits[..., 1] = -80.
        tiny = torch.finfo(torch.float32).tiny
        with torch.no_grad(), mock.patch.object(self.model, 'features', return_value=hidden), \
                mock.patch.object(self.model.head, 'forward', return_value=logits):
            for strength in (0., .3, 1e-35):
                self.model.config['cache_weight'] = strength
                probabilities = self.model.predict_probabilities(ids)
                log_probs = self.model.predict_log_probs(ids)
                self.assertTrue(torch.isfinite(log_probs).all())
                torch.testing.assert_close(probabilities, log_probs.exp(), atol=2*tiny, rtol=2e-5)
                torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, 6), atol=1e-6, rtol=0.)


if __name__ == '__main__':
    unittest.main()
