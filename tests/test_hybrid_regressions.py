"""Synthetic regression tests for the hybrid predictor and the count model.

The tensor-mixture tests exercise the cache and count mixing path on repeated
token patterns (causality, padding, chunking and batch independence). The MKN
test compares the packed-array estimator in fit_ngram.py with a separate
tuple/dictionary implementation on a synthetic Markov source, for both pruned
and unpruned counts. No corpus or checkpoint is loaded.
"""
from collections import Counter, defaultdict
from pathlib import Path
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import hybrid
import student
from common import windows
from evaluate import score
from fit_ngram import VOCAB, fit
from ngram_expert import NGramExpert


def synthetic_tokens(n=40000, seed=7):
    rng = np.random.default_rng(seed)
    active = rng.choice(VOCAB, size=60, replace=False)
    transitions = {}
    result = [int(active[0]), int(active[1])]
    for _ in range(n - 2):
        key = tuple(result[-2:])
        if key not in transitions:
            support = rng.choice(active, size=rng.integers(2, 8), replace=False)
            transitions[key] = support, rng.dirichlet(np.full(len(support), .5))
        support, weights = transitions[key]
        token = rng.choice(active) if rng.random() < .08 else rng.choice(support, p=weights)
        result.append(int(token))
    return np.asarray(result, dtype=np.int64)


def dictionary_mkn(tokens, max_order=5, min_count=2):
    """Tuple/dictionary estimator, independent of packed-key array operations."""
    raw = {n: Counter(tuple(map(int, tokens[i:i+n]))
                      for i in range(len(tokens)-n+1))
           for n in range(2, max_order+1)}
    unigram = np.full(VOCAB, .1, dtype=np.float64)
    for gram in raw[2]:
        unigram[gram[-1]] += 1
    unigram /= unigram.sum()
    direct, backoff, statistics = {}, {}, {}
    for order in range(2, max_order+1):
        counts = raw[order] if order == max_order else Counter(g[1:] for g in raw[order+1])
        frequencies = [sum(c == k for c in counts.values()) for k in range(1, 5)]
        n1, n2, n3, n4 = frequencies
        y = n1/(n1+2*n2)
        discounts = [1-2*y*n2/n1, 2-3*y*n3/n2, 3-4*y*n4/n3]
        totals = defaultdict(int)
        for gram, count in counts.items():
            totals[gram[:-1]] += count
        direct[order], mass = defaultdict(dict), defaultdict(float)
        for gram, count in counts.items():
            if count < (1 if order == 2 else min_count):
                continue
            value = max(count-discounts[min(count, 3)-1], 0.)/totals[gram[:-1]]
            if value:
                direct[order][gram[:-1]][gram[-1]] = value
                mass[gram[:-1]] += value
        backoff[order] = {context: 1-value for context, value in mass.items()}
        statistics[order] = (frequencies, discounts, len(counts))

    def probability(history):
        result = unigram.copy()
        for order in range(2, min(max_order, len(history)+1)+1):
            context = tuple(history[-(order-1):])
            result *= backoff[order].get(context, 1.)
            for token, mass in direct[order].get(context, {}).items():
                result[token] += mass
        return result

    return probability, direct, statistics


class HybridRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='mp1-hybrid-', dir=os.environ.get('MP1_CHECK_TEMP'))
        cls.tokens = synthetic_tokens()
        cls.arrays, cls.statistics = fit(cls.tokens, 5, 2)
        cls.asset = Path(cls.temp.name)/'synthetic-mkn.npz'
        np.savez(cls.asset, **cls.arrays)
        cls.config = dict(vocab=2048, width=32, heads=4, depth=2, mlp_hidden=64,
                          context=256, dropout=0., embedding_dropout=0., attention_dropout=0.,
                          logit_temperature=1.025, cache_weight=.15, cache_temperature=12.,
                          cache_min_history=16, cache_min_similarity=.75, cache_layer=-1,
                          cache_decay=0., ngram_weight=.075, ngram_early_weight=.2,
                          ngram_early_tokens=16, ngram_confidence_power=0.,
                          ngram_asset='synthetic-mkn.npz', cpu_batch_chunk=0)
        with torch.random.fork_rng(), mock.patch('hybrid.verified_asset', return_value=cls.asset):
            torch.manual_seed(1742)
            cls.model = hybrid.HybridGPT(cls.config).eval()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.model.config = dict(self.config)

    def repeated_inputs(self, length=48):
        # Distinct rows and short periodic fragments create repeated token states.
        # The real cache routine still computes similarities/gates without mocks.
        rows = []
        for row in range(10):
            fragment = self.tokens[100+row*13:100+row*13+2+row % 5]
            rows.append(np.resize(fragment, length))
        return torch.tensor(np.stack(rows), dtype=torch.long)

    def observed_prediction(self, ids):
        cache_records, coefficient_records = [], []
        original_cache = student.neural_cache_distribution
        original_coefficient = hybrid.count_mixing_coefficients

        def observe_cache(*args, **kwargs):
            weights, active = original_cache(*args, **kwargs)
            cache_records.append(active.expand(args[0].shape[0], -1, -1).detach().clone())
            return weights, active

        def observe_coefficient(*args, **kwargs):
            value = original_coefficient(*args, **kwargs)
            coefficient_records.append(value)
            return value

        with torch.no_grad(), mock.patch('student.neural_cache_distribution', side_effect=observe_cache), \
                mock.patch('hybrid.count_mixing_coefficients', side_effect=observe_coefficient):
            result = self.model.predict_log_probs(ids)
        self.assertTrue(cache_records and coefficient_records)
        active = torch.cat(cache_records)
        self.assertFalse(active[:, :16].any())
        if ids.shape[1] > 16:
            self.assertTrue(active[:, 16:].any())
        for coefficient in coefficient_records:
            self.assertIsInstance(coefficient, torch.Tensor, 'Must execute tensor-coefficient branch.')
            self.assertTrue(torch.all(coefficient[:, :16] == .2))
            self.assertTrue(torch.all(coefficient[:, 16:] == .075))
        return result, active

    def test_tensor_mixture_has_active_cache_and_normalized_finite_output(self):
        ids = self.repeated_inputs()
        actual, active = self.observed_prediction(ids)
        self.assertTrue(active[:, 16:].flatten(1).any(1).all(), 'Every synthetic row should exercise copying.')
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLessEqual(float(actual.logsumexp(-1).abs().max()), 1e-6)
        self.assertLessEqual(float((actual.exp().sum(-1)-1).abs().max()), 1e-6)
        self.model.config['cache_weight'] = 0.
        with torch.no_grad():
            no_cache = self.model.predict_log_probs(ids)
        self.assertGreater(float((actual[:, 16:]-no_cache[:, 16:]).abs().max()), .01)

    def test_tensor_mixture_suffix_perturbations_preserve_prefixes(self):
        ids = self.repeated_inputs()
        original, _ = self.observed_prediction(ids)
        for cut in [8, 15, 16, 17, 30]:
            with self.subTest(prefix_length=cut):
                changed = ids.clone()
                changed[:, cut:] = (changed[:, cut:]+137) % VOCAB
                actual, _ = self.observed_prediction(changed)
                torch.testing.assert_close(actual[:, :cut], original[:, :cut], atol=1e-6, rtol=0.)

    def test_tensor_mixture_zero_padding_equals_exact_prefix(self):
        ids = self.repeated_inputs()
        for length in [8, 15, 16, 17, 30, 40]:
            with self.subTest(prefix_length=length):
                padded = torch.zeros_like(ids)
                padded[:, :length] = ids[:, :length]
                exact, _ = self.observed_prediction(ids[:, :length])
                actual, _ = self.observed_prediction(padded)
                torch.testing.assert_close(actual[:, :length], exact, atol=3e-6, rtol=0.)

    def test_tensor_mixture_chunks_4_4_2_and_batch_independence(self):
        ids = self.repeated_inputs()
        expected, _ = self.observed_prediction(ids)
        self.model.config['cpu_batch_chunk'] = 4
        original = self.model._predict_unchunked
        sizes = []

        def observe_chunk(part):
            sizes.append(part.shape[0])
            return original(part)

        with mock.patch.object(self.model, '_predict_unchunked', side_effect=observe_chunk):
            actual, _ = self.observed_prediction(ids)
        self.assertEqual(sizes, [4, 4, 2])
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=0.)
        with torch.no_grad():
            reverse = self.model.predict_log_probs(ids.flip(0)).flip(0)
            single = torch.cat([self.model.predict_log_probs(row[None]) for row in ids])
            self.model.predict_log_probs((ids+19) % VOCAB)
            repeated = self.model.predict_log_probs(ids)
        for alternate in [reverse, single, repeated]:
            torch.testing.assert_close(alternate, expected, atol=3e-6, rtol=0.)

    def test_official_scorer_synthetic_ten_row_tail(self):
        tokens = self.repeated_inputs(length=9*256+118)[0]
        batch = list(windows(tokens, 32))
        self.assertEqual(len(batch), 1)
        ids, targets = batch[0]
        self.assertEqual(tuple(ids.shape), (10, 256))
        self.assertEqual(int((targets[-1] != -100).sum()), 117)
        self.model.config['cpu_batch_chunk'] = 4
        _, active = self.observed_prediction(ids)
        self.assertGreater(int(active.sum()), 0)
        together = score(self.model, tokens, len(tokens)*3, torch.device('cpu'), 'fp32', batch_size=32)
        singles = score(self.model, tokens, len(tokens)*3, torch.device('cpu'), 'fp32', batch_size=1)
        self.model.config['cpu_batch_chunk'] = 0
        unchunked = score(self.model, tokens, len(tokens)*3, torch.device('cpu'), 'fp32', batch_size=32)
        self.assertEqual(together['targets'], len(tokens)-1)
        for alternate in [singles, unchunked]:
            np.testing.assert_allclose(together['window_nll_nats'], alternate['window_nll_nats'], atol=1e-4, rtol=0.)

    def test_mkn_dictionary_reference_pruned_and_unpruned(self):
        rng = np.random.default_rng(3)
        rows = [self.tokens[s:s+40] for s in [100, 503, 1117]]
        rows += [synthetic_tokens(40, seed=99), rng.integers(0, VOCAB, 40)]
        ids = torch.tensor(np.stack(rows), dtype=torch.long)
        for minimum in [1, 2]:
            with self.subTest(min_count=minimum):
                arrays, statistics = fit(self.tokens, 5, minimum)
                reference, direct, expected_statistics = dictionary_mkn(self.tokens, 5, minimum)
                path = Path(self.temp.name)/f'mkn-min{minimum}.npz'
                np.savez(path, **arrays)
                expert = NGramExpert(path)
                for stat in statistics:
                    order = stat['order']
                    frequencies, discounts, unpruned_entries = expected_statistics[order]
                    self.assertEqual(stat['count_frequencies_1_to_4'], frequencies)
                    np.testing.assert_allclose(stat['modified_discounts'], discounts, atol=1e-12, rtol=0.)
                    self.assertEqual(stat['entries'], sum(len(values) for values in direct[order].values()))
                    if minimum == 2 and order > 2:
                        self.assertLess(stat['entries'], unpruned_entries)
                expected = np.stack([np.stack([reference(row[:t+1].tolist()) for t in range(40)]) for row in ids])
                with torch.no_grad():
                    actual = expert(ids).double().numpy()
                self.assertTrue(np.isfinite(actual).all() and (actual > 0).all())
                np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=0.)
                np.testing.assert_allclose(actual.sum(-1), 1., atol=1e-6, rtol=0.)


if __name__ == '__main__':
    unittest.main()
