"""Compact rotary transformer with an optional causal, within-window neural cache.

Only ``forward`` participates in language-model training. ``predict_log_probs``
can mix its probabilities with a cache assembled from the current input prefix.
The cache has no learned asset and is reconstructed independently on every call.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class RotaryAttention(nn.Module):
    def __init__(self, width, heads, context, dropout, theta):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError('Attention requires an even head dimension.')
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        inv_frequency = theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        phase = torch.outer(torch.arange(context).float(), inv_frequency)
        self.register_buffer('cos', phase.cos()[None, None], persistent=False)
        self.register_buffer('sin', phase.sin()[None, None], persistent=False)

    def rotate(self, x):
        # Each adjacent coordinate pair represents one complex coordinate.
        cos = self.cos[:, :, :x.shape[-2]].to(dtype=x.dtype)
        sin = self.sin[:, :, :x.shape[-2]].to(dtype=x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.rotate(q), self.rotate(k)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.,
        )
        return self.proj(attended.transpose(1, 2).reshape(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['width']
        norm = RMSNorm if config.get('norm', 'rms') == 'rms' else nn.LayerNorm
        self.norm1, self.norm2 = norm(width), norm(width)
        dropout = float(config.get('dropout', .15))
        self.attn = RotaryAttention(width, config['heads'], config['context'],
                                    float(config.get('attention_dropout', dropout)),
                                    float(config.get('rope_theta', 10000.)))
        hidden = config.get('mlp_hidden')
        if hidden is None:
            hidden = 64 * math.ceil(width * float(config.get('mlp_ratio', 8 / 3)) / 64)
        self.mlp = SwiGLU(width, int(hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.dropout(self.mlp(self.norm2(x)))


def causal_cache_max_similarity(similarity):
    """Maximum raw cosine among keys with observed successors, independently per row."""
    if similarity.shape[-1] == 0:
        return similarity.new_full(similarity.shape[:-1], float('-inf'))
    query = torch.arange(similarity.shape[-2], device=similarity.device)[:, None]
    key = torch.arange(similarity.shape[-1], device=similarity.device)[None, :]
    return similarity.masked_fill(~(key < query)[None], float('-inf')).amax(dim=-1)


def neural_cache_distribution(ids, hidden, temperature=15., decay=0., min_history=1,
                              similarity=None, min_similarity=-1.):
    """Return weights for observed hidden-state/next-token pairs.

    At query position t, key i describes the prefix ending at i and its value is
    ids[i+1]. The strict mask i<t ensures that this value has already been read.
    The final key has no observed successor and is never eligible. Position zero
    receives zero weights and must use only the language model. No tensor or
    cache is retained after this function returns. An optional gate requires a
    sufficiently similar eligible key before cache interpolation is activated.
    Its threshold is applied to raw cosine similarity, before temperature/decay.
    """
    if not -1. <= min_similarity <= 1.:
        raise ValueError('cache_min_similarity must be in [-1, 1].')
    batch, length = ids.shape
    if length < 2:
        return hidden.new_zeros((batch, length, 0), dtype=torch.float32), ids.new_zeros((1, length, 1), dtype=torch.bool)
    if similarity is None:
        normalized = F.normalize(hidden.float(), dim=-1)
        similarity = torch.matmul(normalized, normalized[:, :-1].transpose(-1, -2))
    scores = similarity * temperature
    query = torch.arange(length, device=ids.device)[:, None]
    key = torch.arange(length - 1, device=ids.device)[None, :]
    eligible = key < query
    if decay:
        scores = scores - decay * (query - key).clamp_min(0)
    # Avoid an all-negative-infinity first row; its weights are subsequently zero.
    scores = scores.masked_fill(~eligible[None], torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1) * eligible[None]
    active = (query >= max(1, min_history))[None]
    if min_similarity > -1.:
        active = active & (causal_cache_max_similarity(similarity) >= min_similarity)[..., None]
    return weights, active


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = int(config['context'])
        width, vocab = config['width'], config['vocab']
        self.token = nn.Embedding(vocab, width)
        self.embedding_dropout = nn.Dropout(float(config.get('embedding_dropout', .1)))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['depth'])])
        self.norm = RMSNorm(width) if config.get('norm', 'rms') == 'rms' else nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.token.weight
        self.apply(self.initialize)
        residual_std = .02 / math.sqrt(2 * config['depth'])
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, std=residual_std)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        if ids.shape[1] > self.context:
            raise ValueError(f'Input length exceeds context={self.context}.')
        x = self.embedding_dropout(self.token(ids))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def features_for_cache(self, ids):
        """Return final prediction features and a selected causal cache representation.

        -1 preserves the original final-normalized representation. Positive
        values select the raw output of that one-based transformer block. Both
        tensors are local to this call; the training features() path is unchanged.
        """
        layer = int(self.config.get('cache_layer', -1))
        if layer == -1:
            hidden = self.features(ids)
            return hidden, hidden
        if not 1 <= layer <= len(self.blocks):
            raise ValueError('cache_layer must be -1 or a block number from 1 through depth.')
        if ids.shape[1] > self.context:
            raise ValueError(f'Input length exceeds context={self.context}.')
        x = self.embedding_dropout(self.token(ids))
        cache_hidden = None
        for index, block in enumerate(self.blocks, 1):
            x = block(x)
            if index == layer:
                # Blocks create new residual tensors, so no clone is required.
                cache_hidden = x
        return self.norm(x), cache_hidden

    def forward(self, ids):
        """Unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def _calibrated_logits(self, hidden):
        temperature = float(self.config.get('logit_temperature', 1.))
        if temperature <= 0.:
            raise ValueError('logit_temperature must be positive.')
        logits = self.head(hidden).float()
        if temperature != 1.:
            logits = logits / temperature
        return logits

    def _cache_strength(self):
        strength = float(self.config.get('cache_weight', 0.))
        if not 0. <= strength < 1.:
            raise ValueError('cache_weight must be in [0, 1).')
        return strength

    def _mix_cache_probabilities(self, ids, hidden, probabilities, strength):
        if strength == 0. or ids.shape[1] < 2:
            return probabilities
        weights, active = neural_cache_distribution(
            ids, hidden,
            temperature=float(self.config.get('cache_temperature', 15.)),
            decay=float(self.config.get('cache_decay', 0.)),
            min_history=max(1, int(self.config.get('cache_min_history', 1))),
            min_similarity=float(self.config.get('cache_min_similarity', -1.)),
        )
        values = ids[:, None, 1:].expand(-1, ids.shape[1], -1)
        mixing = active.to(probabilities.dtype) * strength
        # Scatter the sparse cache mass directly into the neural probabilities.
        # This avoids a second dense vocabulary tensor and dense logaddexp.
        mixed = probabilities * (1 - mixing)
        mixed.scatter_add_(-1, values, weights * mixing)
        return mixed

    def predict_probabilities(self, ids):
        """FP32 probabilities for downstream mixtures, with no dense log/exp round trip.

        Extremely unlikely events can underflow to zero in this probability API.
        The public log-probability scorer interface below remains finite.
        """
        hidden, cache_hidden = self.features_for_cache(ids)
        probabilities = F.softmax(self._calibrated_logits(hidden), dim=-1)
        return self._mix_cache_probabilities(ids, cache_hidden, probabilities, self._cache_strength())

    def predict_log_probs(self, ids):
        hidden, cache_hidden = self.features_for_cache(ids)
        log_probs = F.log_softmax(self._calibrated_logits(hidden), dim=-1)
        strength = self._cache_strength()
        if strength == 0. or ids.shape[1] < 2:
            return log_probs
        mixed = self._mix_cache_probabilities(ids, cache_hidden, log_probs.exp(), strength)
        return mixed.clamp_min(torch.finfo(torch.float32).tiny).log()


def build_model(config):
    return StudentGPT(config)
