"""Hybrid inference contracts and aggregate tuning equivalence."""
import unittest
from unittest import mock

import torch

from common import ROOT, sha
from hybrid import HybridGPT, count_mixing_coefficients
from student import StudentGPT
from tune_cache import target_base_probabilities, target_cache_probabilities
from tune_hybrid import accumulate_hybrid_nll


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.config = dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                          dropout=0., attention_dropout=0., embedding_dropout=0.,
                          ngram_asset='assets/ngram_mkn5.npz',
                          ngram_sha256=sha(ROOT/'assets/ngram_mkn5.npz'),
                          ngram_weight=.3, cache_weight=.1, cache_temperature=15.,
                          cache_decay=0., cache_min_history=1, logit_temperature=1.)
        cls.model = HybridGPT(cls.config).eval()

    def setUp(self):
        self.model.config = dict(self.config, cpu_batch_chunk=0)

    def test_default_zero_weight_and_state_compatibility(self):
        config = dict(vocab=2048, width=32, heads=4, depth=2, context=256)
        plain = StudentGPT(config).eval()
        hybrid = HybridGPT(config).eval()
        hybrid.load_state_dict(plain.state_dict(), strict=True)
        self.assertEqual(list(plain.state_dict()), list(hybrid.state_dict()))
        self.assertEqual(list(plain.state_dict()), list(self.model.state_dict()))
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            torch.testing.assert_close(plain.predict_log_probs(ids), hybrid.predict_log_probs(ids))

    def test_direct_probability_mixture_matches_previous_log_exp_path(self):
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            counts = self.model.ngram_expert(ids)
            for cache_weight in [0., .2]:
                self.model.config.update(cache_weight=cache_weight, cache_min_similarity=.7)
                for ngram_weight in [.15, .9]:
                    self.model.config['ngram_weight'] = ngram_weight
                    old_neural = StudentGPT.predict_log_probs(self.model, ids).exp()
                    expected = ((1-ngram_weight)*old_neural + ngram_weight*counts).clamp_min(torch.finfo(torch.float32).tiny).log()
                    actual = self.model.predict_log_probs(ids)
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-6)
            hidden = torch.randn(2, 12, 32)
            logits = torch.linspace(-1000., 0., 2048).expand(2, 12, -1).clone()
            self.model.config.update(ngram_weight=.15, cache_weight=.2)
            with mock.patch.object(self.model, 'features', return_value=hidden), mock.patch.object(self.model.head, 'forward', return_value=logits):
                old_neural = StudentGPT.predict_log_probs(self.model, ids).exp()
                expected = (.85*old_neural+.15*counts).clamp_min(torch.finfo(torch.float32).tiny).log()
                actual = self.model.predict_log_probs(ids)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-6)
                torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2, 12), atol=1e-6, rtol=1e-6)

    def test_short_inputs_prefixes_and_normalization(self):
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            full = self.model.predict_log_probs(ids)
            for length in [1, 2, 3, 4, 5, 12]:
                short = self.model.predict_log_probs(ids[:, :length])
                self.assertTrue(torch.isfinite(short).all())
                torch.testing.assert_close(short.logsumexp(-1), torch.zeros(2, length), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(full[:, :length], short, atol=2e-5, rtol=1e-6)
            changed = ids.clone()
            changed[:, 7:] = (changed[:, 7:]+31) % 2048
            torch.testing.assert_close(full[:, :7], self.model.predict_log_probs(changed)[:, :7], atol=1e-6, rtol=1e-6)

    def test_batch_independence_and_cpu_chunks(self):
        ids = torch.randint(0, 2048, (5, 12))
        with torch.no_grad():
            together = self.model.predict_log_probs(ids)
            first = self.model.predict_log_probs(ids[:1])
            self.model.predict_log_probs((ids+19) % 2048)
            torch.testing.assert_close(first, self.model.predict_log_probs(ids[:1]), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(together[:1], first, atol=2e-5, rtol=1e-6)
            for chunk in [1, 2, 4, 8]:
                self.model.config['cpu_batch_chunk'] = chunk
                torch.testing.assert_close(together, self.model.predict_log_probs(ids), atol=2e-5, rtol=1e-6)

    def test_forward_remains_differentiable(self):
        self.model.zero_grad(set_to_none=True)
        batch = torch.randint(0, 2048, (2, 13))
        loss = torch.nn.functional.cross_entropy(self.model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_inference_storage_reuse_preserves_gradient_enabled_path(self):
        self.model.zero_grad(set_to_none=True)
        ids = torch.randint(0, 2048, (2, 12))
        differentiable = self.model.predict_log_probs(ids)
        with torch.no_grad():
            inference = self.model.predict_log_probs(ids)
        torch.testing.assert_close(differentiable.detach(), inference, atol=0., rtol=0.)
        (-differentiable[..., 19].mean()).backward()
        grads = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(gradient).all() for gradient in grads))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in grads), 0.)

    def test_asset_hash_and_path_are_enforced(self):
        with self.assertRaises(ValueError):
            HybridGPT(dict(self.config, ngram_sha256='incorrect'))
        with self.assertRaises(ValueError):
            HybridGPT(dict(self.config, ngram_asset='../outside.npz'))

    def test_count_confidence_and_early_position_formula(self):
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            probs, confidence = self.model.ngram_expert.probabilities_and_confidence(ids)
            torch.testing.assert_close(probs, self.model.ngram_expert(ids), atol=0, rtol=0)
            short_probs, short_confidence = self.model.ngram_expert.probabilities_and_confidence(ids[:, :3])
            torch.testing.assert_close(probs[:, :3], short_probs, atol=0, rtol=0)
            torch.testing.assert_close(confidence[:, :3], short_confidence, atol=0, rtol=0)
            self.assertTrue(((confidence >= 0) & (confidence <= 1)).all())
            setting = dict(ngram_weight=.2, ngram_early_weight=.5, ngram_early_tokens=3, ngram_confidence_power=1.)
            coefficient = count_mixing_coefficients(setting, ids, confidence)
            expected = confidence*.2
            expected[:, :3] = confidence[:, :3]*.5
            torch.testing.assert_close(coefficient, expected)
            self.model.config.update(setting)
            mixed = self.model.predict_log_probs(ids)
            torch.testing.assert_close(mixed.logsumexp(-1), torch.zeros(2, 12), atol=1e-6, rtol=1e-6)
            self.model.config['cpu_batch_chunk'] = 1
            torch.testing.assert_close(mixed, self.model.predict_log_probs(ids), atol=2e-5, rtol=1e-6)

    def test_aggregate_with_dynamic_count_weights(self):
        batch = torch.randint(0, 2048, (2, 13))
        ids, targets = batch[:, :-1], batch[:, 1:]
        valid = torch.ones_like(targets, dtype=torch.bool)
        settings = [dict(ngram_weight=.2, ngram_early_weight=.5, ngram_early_tokens=3, ngram_confidence_power=0.),
                    dict(ngram_weight=.4, ngram_early_weight=None, ngram_early_tokens=0, ngram_confidence_power=1.)]
        with torch.no_grad():
            hidden = self.model.features(ids)
            base = target_base_probabilities(self.model.head(hidden).float(), targets, [1.])
            cache, active = target_cache_probabilities(ids, hidden, targets, [(15., 0., 1, .8)])
            full_count, confidence = self.model.ngram_expert.probabilities_and_confidence(ids)
            count = full_count.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            coefficients = torch.stack([count_mixing_coefficients(s, ids, confidence) for s in settings])
            aggregate = accumulate_hybrid_nll(base, cache, active, count, [.1], [.2, .4], valid, coefficients)
            for index, setting in enumerate(settings):
                self.model.config.update(setting, cache_weight=.1, cache_min_similarity=.8)
                actual = -self.model.predict_log_probs(ids).gather(-1, targets.unsqueeze(-1)).double().sum()
                torch.testing.assert_close(actual, aggregate[0, 0, 0, index], atol=2e-5, rtol=1e-7)

    def test_selected_cache_layer_aggregate_and_chunk_agreement(self):
        batch = torch.randint(0, 2048, (3, 13))
        ids, targets = batch[:, :-1], batch[:, 1:]
        valid = torch.ones_like(targets, dtype=torch.bool)
        with torch.no_grad():
            count = self.model.ngram_expert(ids).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for layer in [-1, 1, 2]:
                self.model.config.update(cache_layer=layer, cache_weight=.2,
                                         cache_temperature=12., cache_min_similarity=.7,
                                         cpu_batch_chunk=0)
                final, cache_hidden = self.model.features_for_cache(ids)
                base = target_base_probabilities(self.model.head(final).float(), targets, [1.])
                cache, active = target_cache_probabilities(ids, cache_hidden, targets, [(12., 0., 1, .7)])
                aggregate = accumulate_hybrid_nll(base, cache, active, count, [.2], [.3], valid)
                logp = self.model.predict_log_probs(ids)
                actual = -logp.gather(-1, targets.unsqueeze(-1)).double().sum()
                torch.testing.assert_close(actual, aggregate[0, 0, 0, 0], atol=2e-5, rtol=1e-7)
                self.model.config['cpu_batch_chunk'] = 1
                torch.testing.assert_close(logp, self.model.predict_log_probs(ids), atol=2e-5, rtol=1e-6)

    def test_aggregate_sweep_matches_full_predictor(self):
        batch = torch.randint(0, 2048, (2, 13))
        ids, targets = batch[:, :-1], batch[:, 1:].clone()
        targets[-1, -3:] = -100
        valid = targets != -100
        temperatures = [.9, 1.1]
        cache_settings = [(8., .01, 1, -1.), (15., 0., 8, .9)]
        cache_weights, count_weights = [0., .2], [0., .3]
        with torch.no_grad():
            hidden = self.model.features(ids)
            base = target_base_probabilities(self.model.head(hidden).float(), targets, temperatures)
            cache, active = target_cache_probabilities(ids, hidden, targets, cache_settings)
            count = self.model.ngram_expert(ids).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            aggregate = accumulate_hybrid_nll(base, cache, active, count, cache_weights, count_weights, valid)
            for i, temperature in enumerate(temperatures):
                for j, (cache_temperature, decay, minimum, similarity) in enumerate(cache_settings):
                    for k, cache_weight in enumerate(cache_weights):
                        for m, count_weight in enumerate(count_weights):
                            self.model.config.update(logit_temperature=temperature, cache_temperature=cache_temperature,
                                                     cache_decay=decay, cache_min_history=minimum,
                                                     cache_min_similarity=similarity,
                                                     cache_weight=cache_weight, ngram_weight=count_weight)
                            logp = self.model.predict_log_probs(ids)
                            nll = -logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                            actual = nll.masked_fill(~valid, 0.).double().sum()
                            torch.testing.assert_close(actual, aggregate[i, j, k, m], atol=2e-5, rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
