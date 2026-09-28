"""Student transformer with a compact, training-derived n-gram expert.

Neural parameter names are unchanged, so a student checkpoint loads directly.
The separate count asset contains no validation or test statistics. It is read
once at construction, verified by SHA-256, and never updated during inference.
"""
from pathlib import Path

import torch
from torch.nn import functional as F

from common import ROOT, sha
from ngram_expert import NGramExpert
from student import StudentGPT, neural_cache_distribution


def count_mixing_coefficients(config, ids, confidence=None):
    """Fixed formula with validation-selected scalars and train-derived mass."""
    weight = float(config.get('ngram_weight', 0.))
    early = config.get('ngram_early_weight')
    early = weight if early is None else float(early)
    cutoff = int(config.get('ngram_early_tokens', 0))
    power = float(config.get('ngram_confidence_power', 0.))
    if not 0 <= weight < 1 or not 0 <= early < 1 or cutoff < 0 or power < 0:
        raise ValueError('Invalid count mixture weight, early cutoff, or confidence power.')
    if (not cutoff or early == weight) and not power:
        return weight
    positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
    coefficient = torch.where(positions < cutoff, early, weight).expand(ids.shape[0], -1)
    if power:
        if confidence is None:
            raise ValueError('Positive confidence power requires training-derived context mass.')
        coefficient = coefficient * confidence.pow(power)
    return coefficient


def verified_asset(config):
    relative = Path(config['ngram_asset'])
    if relative.is_absolute():
        raise ValueError('ngram_asset must be relative to the submitted code directory.')
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('ngram_asset must stay inside the submitted code directory.')
    expected = config.get('ngram_sha256')
    if not expected or sha(path) != expected:
        raise ValueError('Missing or mismatched SHA-256 for n-gram inference asset.')
    return path


TINY = torch.finfo(torch.float32).tiny


def count_components(expert, ids):
    """Sparse/backoff decomposition of the count mixture; same table lookups."""
    batch, length = ids.shape
    size = batch*length
    history = ids.clone()
    backoffs, sparse = [], []
    for order in expert.orders:
        if order == 2:
            continue
        h = order-1
        if length < h:
            break
        older = torch.zeros_like(ids)
        older[:, h-1:] = ids[:, :length-h+1]
        history = older*(2048**(h-1))+history
        query = history.reshape(-1).contiguous()
        contexts = getattr(expert, f'contexts_{order}')
        if contexts.numel() == 0:
            continue
        lookup = torch.searchsorted(contexts, query)
        safe = lookup.clamp_max(len(contexts)-1)
        present = (lookup < len(contexts)) & (contexts[safe] == query)
        present &= torch.arange(length, device=ids.device).repeat(batch) >= h-1
        backoffs.append(torch.where(present, getattr(expert, f'backoff_{order}')[safe], 1.))
        starts = getattr(expert, f'starts_{order}')[safe]
        counts = torch.where(present, getattr(expert, f'lengths_{order}')[safe], 0)
        rows = torch.repeat_interleave(torch.arange(size, device=ids.device), counts)
        n = rows.numel()
        if n:
            offsets = torch.repeat_interleave(counts.cumsum(0)-counts, counts)
            entries = torch.repeat_interleave(starts, counts)+torch.arange(n, device=ids.device)-offsets
            sparse.append((len(backoffs)-1, rows, getattr(expert, f'keys_{order}')[entries]%2048,
                           getattr(expert, f'direct_{order}')[entries]))
    return backoffs, sparse


class HybridGPT(StudentGPT):
    def __init__(self, config):
        super().__init__(config)
        weight = float(config.get('ngram_weight', 0.))
        if not 0. <= weight < 1.:
            raise ValueError('ngram_weight must be in [0, 1).')
        if int(config.get('cpu_batch_chunk', 0)) < 0:
            raise ValueError('cpu_batch_chunk must be nonnegative.')
        self.ngram_expert = None
        if 'ngram_asset' in config:
            self.ngram_expert = NGramExpert(verified_asset(config))
        elif weight or float(config.get('ngram_early_weight') or 0.):
            raise ValueError('A positive ngram_weight requires its asset and SHA-256.')

    def _predict_original(self, ids):
        chunk = int(self.config.get('cpu_batch_chunk', 0))
        if chunk < 0:
            raise ValueError('cpu_batch_chunk must be nonnegative.')
        if ids.device.type == 'cpu' and chunk and ids.shape[0] > chunk:
            return torch.cat([self._predict_unchunked(part) for part in ids.split(chunk, dim=0)], dim=0)
        return self._predict_unchunked(ids)

    def _predict_unchunked(self, ids):
        weight = float(self.config.get('ngram_weight', 0.))
        if not 0. <= weight < 1.:
            raise ValueError('ngram_weight must be in [0, 1).')
        early = float(self.config.get('ngram_early_weight') or 0.)
        if weight == 0. and early == 0.:
            return super().predict_log_probs(ids)
        if self.ngram_expert is None:
            raise ValueError('N-gram inference asset was not loaded.')
        neural = super().predict_probabilities(ids)
        if float(self.config.get('ngram_confidence_power', 0.)):
            count_probs, confidence = self.ngram_expert.probabilities_and_confidence(ids)
        else:
            count_probs, confidence = self.ngram_expert(ids), None
        mixing = count_mixing_coefficients(self.config, ids, confidence)
        # The count expert has positive probability for every token. One final
        # logarithm avoids the extra dense logarithm/logaddexp work on CPU.
        if isinstance(mixing, float):
            if not torch.is_grad_enabled():
                # These probabilities are fresh local tensors. Reuse storage
                # during inference; retain the ordinary path when gradients are
                # enabled so autograd's saved values cannot be overwritten.
                neural.mul_(1-mixing).add_(count_probs, alpha=mixing)
                return neural.clamp_min_(torch.finfo(torch.float32).tiny).log_()
            mixed = torch.add(neural * (1-mixing), count_probs, alpha=mixing)
        else:
            mixed = neural * (1-mixing[..., None]) + count_probs * mixing[..., None]
        return mixed.clamp_min(torch.finfo(torch.float32).tiny).log()


    def predict_log_probs(self, ids):
        chunk = int(self.config.get('cpu_batch_chunk', 0))
        if chunk < 0:
            raise ValueError('cpu_batch_chunk must be nonnegative.')
        trunk = self.config.get('cpu_trunk_chunk', 0)
        if type(trunk) is not int or trunk < 0:
            raise ValueError('cpu_trunk_chunk must be a nonnegative integer.')
        if (not trunk or not chunk or self.training or torch.is_grad_enabled() or
                ids.device.type != 'cpu' or torch.is_autocast_enabled('cpu')):
            return self._predict_original(ids)
        if trunk % chunk:
            raise ValueError('cpu_trunk_chunk must be a positive multiple of cpu_batch_chunk.')
        if float(self.config.get('logit_temperature', 1.)) <= 0.:
            raise ValueError('logit_temperature must be positive.')
        batch, length = ids.shape
        out = torch.empty(batch, length, 2048, dtype=torch.float32, device=ids.device)
        for t0 in range(0, batch, trunk):
            tids = ids[t0:t0+trunk]
            hidden, cache_hidden = self.features_for_cache(tids)
            for v0 in range(0, tids.shape[0], chunk):
                end = min(v0+chunk, tids.shape[0])
                self._vocab_part(tids[v0:end], hidden[v0:end], cache_hidden[v0:end], out[t0+v0:t0+end])
        return out

    def _cache_in_place(self, ids, cache_hidden, probs, strength):
        if strength == 0. or ids.shape[1] < 2:
            return probs
        weights, active = neural_cache_distribution(ids, cache_hidden,
            temperature=float(self.config.get('cache_temperature', 15.)),
            decay=float(self.config.get('cache_decay', 0.)),
            min_history=max(1, int(self.config.get('cache_min_history', 1))),
            min_similarity=float(self.config.get('cache_min_similarity', -1.)))
        values = ids[:, None, 1:].expand(-1, ids.shape[1], -1)
        mixing = active.to(probs.dtype)*strength
        probs.mul_(1-mixing)
        probs.scatter_add_(-1, values, weights*mixing)
        return probs

    def _vocab_part(self, ids, hidden, cache_hidden, out):
        weight = float(self.config.get('ngram_weight', 0.))
        if not 0. <= weight < 1.:
            raise ValueError('ngram_weight must be in [0, 1).')
        early = float(self.config.get('ngram_early_weight') or 0.)
        strength = self._cache_strength()
        temperature = float(self.config.get('logit_temperature', 1.))
        logits = self.head(hidden).float()
        if temperature != 1.:
            logits.div_(temperature)
        if weight == 0. and early == 0.:
            log_probs = F.log_softmax(logits, dim=-1)
            if strength == 0. or ids.shape[1] < 2:
                out.copy_(log_probs)
            else:
                mixed = self._cache_in_place(ids, cache_hidden, log_probs.exp(), strength)
                torch.log(mixed.clamp_min_(TINY), out=out)
            return
        if self.ngram_expert is None:
            raise ValueError('N-gram inference asset was not loaded.')
        neural = self._cache_in_place(ids, cache_hidden, F.softmax(logits, dim=-1), strength)
        if float(self.config.get('ngram_confidence_power', 0.)):
            count_probs, confidence = self.ngram_expert.probabilities_and_confidence(ids)
            mixing = count_mixing_coefficients(self.config, ids, confidence)
        else:
            mixing = count_mixing_coefficients(self.config, ids)
            if self.config.get('cpu_restructure_counts', False):
                self._mix_restructured(ids, neural, mixing)
                torch.log(neural.clamp_min_(TINY), out=out)
                return
            count_probs = self.ngram_expert(ids)
        if isinstance(mixing, float):
            neural.mul_(1-mixing).add_(count_probs, alpha=mixing)
        else:
            neural.mul_(1-mixing[..., None])
            count_probs.mul_(mixing[..., None])
            neural.add_(count_probs)
        torch.log(neural.clamp_min_(TINY), out=out)

    def _mix_restructured(self, ids, neural, mixing):
        expert = self.ngram_expert
        if 2 not in expert.orders:
            raise ValueError('Count restructuring requires the existing bigram table.')
        size = ids.numel()
        flat = neural.view(size, 2048)
        m = torch.full((size,), mixing, device=ids.device) if isinstance(mixing, float) else mixing.reshape(-1).float()
        backoffs, sparse = count_components(expert, ids)
        suffix = [None]*len(backoffs)
        running = torch.ones(size, device=ids.device)
        for k in range(len(backoffs)-1, -1, -1):
            suffix[k] = running
            running = running*backoffs[k]
        flat.mul_((1-m)[:, None])
        flat.addcmul_(expert.bigram[ids.reshape(-1)], (m*running)[:, None])
        for k, rows, token, direct in sparse:
            flat.index_put_((rows, token), direct*(m*suffix[k])[rows], accumulate=True)


def build_model(config):
    return HybridGPT(config)
