"""Frozen two-model probability ensemble for training-time distillation only."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from student import StudentGPT


class TeacherEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = 256
        if not config.get('training_only') or config.get('member_implementations') != ['student', 'student']:
            raise ValueError('Use an explicitly training-only ensemble of two student models.')
        if len(config['member_configs']) != 2:
            raise ValueError('Exactly two member configurations are required.')
        if config.get('vocab') != 2048 or config.get('context') != 256:
            raise ValueError('The teacher must use the fixed vocabulary and context.')
        if any(member.get('vocab') != 2048 or member.get('context') != 256 for member in config['member_configs']):
            raise ValueError('Every member must use the fixed vocabulary and context.')
        self.members = nn.ModuleList([StudentGPT(member) for member in config['member_configs']])
        self.requires_grad_(False)
        self.eval()
        self.settings()

    def settings(self):
        temperature = float(self.config['common_temperature'])
        alpha = float(self.config['alpha'])
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('Temperature must be positive and alpha must be in [0,1].')
        return temperature, alpha

    def train(self, mode=True):
        # This module is a fixed target generator, including when a caller
        # accidentally requests train mode. Student training remains separate.
        return super().train(False)

    @torch.no_grad()
    def forward(self, ids):
        temperature, alpha = self.settings()
        if alpha == 1.:
            return F.log_softmax(self.members[0](ids).float()/temperature, dim=-1)
        if alpha == 0.:
            return F.log_softmax(self.members[1](ids).float()/temperature, dim=-1)
        first = F.log_softmax(self.members[0](ids).float()/temperature, dim=-1)
        second = F.log_softmax(self.members[1](ids).float()/temperature, dim=-1)
        return torch.logaddexp(first+math.log(alpha), second+math.log1p(-alpha))

    def predict_log_probs(self, ids):
        return self(ids)


def build_model(config):
    return TeacherEnsemble(config)
