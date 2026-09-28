"""Compact train-derived n-gram probability expert for independent input windows.

Load assets written by fit_ngram.py. The helper returns full normalized
probability distributions without modifying the benchmark or evaluator.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn


class NGramExpert(nn.Module):
    def __init__(self, asset_path):
        super().__init__()
        self.context = 256
        with np.load(Path(asset_path), allow_pickle=False) as arrays:
            unigram = torch.from_numpy(arrays['unigram'].copy())
            # The NPZ is the single inference asset; avoid duplicating all count
            # arrays into a neural checkpoint that contains this helper.
            self.register_buffer('unigram', unigram, persistent=False)
            orders = sorted({int(key.split('_')[0][1:]) for key in arrays if key.startswith('n')})
            if not orders or orders != list(range(2, max(orders)+1)) or max(orders) > 5:
                raise ValueError('Asset must contain consecutive orders 2 through at most 5.')
            self.orders = orders
            for order in orders:
                keys = arrays[f'n{order}_keys'].copy()
                contexts = arrays[f'n{order}_contexts'].copy()
                self.register_buffer(f'keys_{order}', torch.from_numpy(keys), persistent=False)
                self.register_buffer(f'direct_{order}', torch.from_numpy(arrays[f'n{order}_direct'].copy()), persistent=False)
                self.register_buffer(f'contexts_{order}', torch.from_numpy(contexts), persistent=False)
                self.register_buffer(f'backoff_{order}', torch.from_numpy(arrays[f'n{order}_backoff'].copy()), persistent=False)
                starts = np.searchsorted(keys // 2048, contexts, side='left')
                ends = np.searchsorted(keys // 2048, contexts, side='right')
                self.register_buffer(f'starts_{order}', torch.from_numpy(starts), persistent=False)
                self.register_buffer(f'lengths_{order}', torch.from_numpy(ends-starts), persistent=False)
            if 2 in orders:
                bigram = unigram.expand(2048, -1).clone()
                bigram[self.contexts_2] *= self.backoff_2[:, None]
                bigram[self.keys_2 // 2048, self.keys_2 % 2048] += self.direct_2
                self.register_buffer('bigram', bigram, persistent=False)
                confidence = unigram.new_zeros(2048)
                confidence[self.contexts_2] = 1.-self.backoff_2
                self.register_buffer('bigram_confidence', confidence, persistent=False)

    @torch.no_grad()
    def forward(self, ids):
        return self._predict(ids, return_confidence=False)

    @torch.no_grad()
    def probabilities_and_confidence(self, ids):
        """Also return retained probability mass in the longest matched context."""
        return self._predict(ids, return_confidence=True)

    def _predict(self, ids, return_confidence):
        batch, length = ids.shape
        size = batch * length
        probs = self.bigram[ids].reshape(size, 2048) if 2 in self.orders else self.unigram.expand(size, -1).clone()
        if return_confidence:
            confidence = self.bigram_confidence[ids].reshape(-1).clone() if 2 in self.orders else self.unigram.new_zeros(size)
        history = ids.clone()
        for order in self.orders:
            if order == 2:
                continue
            h = order - 1
            if length < h:
                break
            # Roll to add precisely the next older token. Left padding is never
            # consulted because positions lacking h observed tokens are masked.
            older = torch.zeros_like(ids)
            older[:, h-1:] = ids[:, :length-h+1]
            history = older * (2048 ** (h-1)) + history
            query = history.reshape(-1).contiguous()
            contexts = getattr(self, f'contexts_{order}')
            if contexts.numel() == 0:
                continue
            lookup = torch.searchsorted(contexts, query)
            safe = lookup.clamp_max(len(contexts)-1)
            present = (lookup < len(contexts)) & (contexts[safe] == query)
            present = present & (torch.arange(length, device=ids.device).repeat(batch) >= h-1)
            backoff = torch.where(present, getattr(self, f'backoff_{order}')[safe], 1.)
            if return_confidence:
                confidence = torch.where(present, 1.-backoff, confidence)
            probs *= backoff[:, None]
            starts = getattr(self, f'starts_{order}')[safe]
            counts = torch.where(present, getattr(self, f'lengths_{order}')[safe], 0)
            rows = torch.repeat_interleave(torch.arange(size, device=ids.device), counts)
            n_entries = rows.numel()
            if n_entries:
                offsets = torch.repeat_interleave(counts.cumsum(0)-counts, counts)
                entries = torch.repeat_interleave(starts, counts) + torch.arange(n_entries, device=ids.device)-offsets
                token = getattr(self, f'keys_{order}')[entries] % 2048
                # Each (row, token) pair is unique for this order.
                probs[rows, token] += getattr(self, f'direct_{order}')[entries]
        if return_confidence:
            return probs.reshape(batch, length, 2048), confidence.clamp(0., 1.).reshape(batch, length)
        return probs.reshape(batch, length, 2048)

    def predict_log_probs(self, ids):
        return self(ids).clamp_min(1e-30).log()
