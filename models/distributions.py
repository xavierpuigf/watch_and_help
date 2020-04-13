import math
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from utils.utils_models import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super(FixedCategorical, self).__init__(probs=probs, logits=logits)
        self.original_logits = logits

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()



class ElementWiseCategorical(nn.Module):
    def __init__(self, num_inputs, method='fc'):
        super(ElementWiseCategorical, self).__init__()
        self.method = method
        if self.method == 'fc':
            self.layer = nn.Sequential(
                nn.Linear(num_inputs, num_inputs),
                nn.Relu(),
                nn.Linear(num_inputs, 1)
            )
        if self.method == 'linear':
            self.layer = nn.Linear(num_inputs, 1)

    def update_logs(self, logs):
        return FixedCategorical(logits=logs)

    def forward(self, x, y):
        # x: [batch, dim]
        # y: [batch, num_nodes (padded), dim]
        # n x dim
        x0 = x.unsqueeze(1)
        if self.method == 'dot':
            raise Exception
            logs = torch.bmm(x0, y.transpose(1,2))[:, 0, :]

        elif self.method == 'fc':
            raise Exception
            num_nodes = y.shape[1]
            comb_embed = torch.cat([x0.repeat(1, num_nodes, 1), y], dim=2)
            logs = self.layer(comb_embed).squeeze(-1)

        elif self.method == 'linear':
            logs = self.layer(y).squeeze(-1)

        #mask_char = np.ones(logs.shape)
        # mask_char[:, 0] = 0.
        #logs = logs * torch.Tensor(mask_char).to(logs.device)

        self.original_logits = logs
        return FixedCategorical(logits=logs)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def update_logs(self, logs):
        return FixedCategorical(logits=logs)

    def forward(self, x):
        x = self.linear(x)
        self.original_logits = x
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
