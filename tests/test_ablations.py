"""Mechanism ablations must leave unrelated selected settings unchanged."""
import unittest

import torch

from ablate_hybrid import ablation_variants
from hybrid import count_mixing_coefficients


class AblationTests(unittest.TestCase):
    def test_early_weight_ablation_retains_fixed_base_and_other_settings(self):
        selected = dict(ngram_weight=.1, ngram_early_weight=.3, ngram_early_tokens=8,
                        ngram_confidence_power=0., cache_weight=.15, cache_temperature=12.,
                        cache_min_history=16, cache_min_similarity=.75, cache_layer=-1,
                        logit_temperature=1.15)
        before = dict(selected)
        variants = dict(ablation_variants(selected))
        self.assertEqual(selected, before)
        self.assertEqual(variants['full_without_early_count_weight'],
                         dict(ngram_early_weight=None, ngram_early_tokens=0))
        self.assertNotIn('full_without_count_confidence', variants)
        removed = dict(selected, **variants['full_without_early_count_weight'])
        ids = torch.zeros(2, 12, dtype=torch.long)
        original = count_mixing_coefficients(selected, ids)
        torch.testing.assert_close(original[:, :8], torch.full((2, 8), .3))
        torch.testing.assert_close(original[:, 8:], torch.full((2, 4), .1))
        self.assertEqual(count_mixing_coefficients(removed, ids), .1)
        for name in selected.keys()-{'ngram_early_weight', 'ngram_early_tokens'}:
            self.assertEqual(removed[name], selected[name])

    def test_confidence_ablation_is_independent_of_early_weight(self):
        selected = dict(ngram_weight=.1, ngram_early_weight=.3, ngram_early_tokens=8,
                        ngram_confidence_power=1., cache_weight=.15)
        variants = dict(ablation_variants(selected))
        self.assertEqual(variants['full_without_count_confidence'], dict(ngram_confidence_power=0.))
        removed = dict(selected, **variants['full_without_count_confidence'])
        self.assertEqual(removed['ngram_early_weight'], .3)
        self.assertEqual(removed['ngram_early_tokens'], 8)
        self.assertEqual(removed['cache_weight'], .15)
        ids = torch.zeros(2, 12, dtype=torch.long)
        confidence = torch.full((2, 12), .5)
        weighted = count_mixing_coefficients(selected, ids, confidence)
        unweighted = count_mixing_coefficients(removed, ids)
        torch.testing.assert_close(weighted*2, unweighted)

    def test_inactive_or_no_effect_count_settings_do_not_add_ablations(self):
        settings = [dict(ngram_weight=.1),
                    dict(ngram_weight=.1, ngram_early_weight=None, ngram_early_tokens=8),
                    dict(ngram_weight=.1, ngram_early_weight=.1, ngram_early_tokens=8),
                    dict(ngram_weight=.1, ngram_early_weight=.3, ngram_early_tokens=0),
                    dict(ngram_weight=0., ngram_confidence_power=1.)]
        for selected in settings:
            variants = dict(ablation_variants(selected))
            self.assertNotIn('full_without_early_count_weight', variants)
            self.assertNotIn('full_without_count_confidence', variants)
            self.assertEqual(variants['full'], {})


if __name__ == '__main__':
    unittest.main()
