import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase


def to_cpu(list_of_tensor):
    if isinstance(list_of_tensor[0], list):
        list_list_of_tensor = list_of_tensor
        list_of_tensor = [to_cpu(list_of_tensor)
                          for list_of_tensor in list_list_of_tensor]
    else:
        list_of_tensor = [tensor.cpu() for tensor in list_of_tensor]
    return list_of_tensor


def average_over_list(l):
    return sum(l) / len(l)


def _LayerNormGRUCell(input, hidden, w_ih, w_hh, ln, b_ih=None, b_hh=None):

    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    # use layernorm here
    resetgate = torch.sigmoid(ln['resetgate'](i_r + h_r))
    inputgate = torch.sigmoid(ln['inputgate'](i_i + h_i))
    newgate = torch.tanh(ln['newgate'](i_n + resetgate * h_n))
    hy = newgate + inputgate * (hidden - newgate)

    return hy


class CombinedEmbedding(nn.Module):

    def __init__(self, pretrained_embedding, embedding):
        super(CombinedEmbedding, self).__init__()
        self.pretrained_embedding = pretrained_embedding
        self.embedding = embedding
        self.pivot = pretrained_embedding.num_embeddings

    def forward(self, input):
        outputs = []
        mask = input < self.pivot
        outputs.append(self.pretrained_embedding(torch.clamp(input, 0, self.pivot-1)) * mask.unsqueeze(1).float())
        mask = input >= self.pivot
        outputs.append(self.embedding(torch.clamp(input, self.pivot) - self.pivot) * mask.unsqueeze(1).float())
        return sum(outputs)


class writer_helper(object):

    def __init__(self, writer):
        self.writer = writer
        self.all_steps = {}

    def get_step(self, tag):
        if tag not in self.all_steps.keys():
            self.all_steps.update({tag: 0})

        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def scalar_summary(self, tag, value, step=None):
        if step is None:
            step = self.get_step(tag)
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step=None):
        if step is None:
            step = self.get_step(tag)
        self.writer.add_text(tag, value, step)


class Constant():
    def __init__(self, v):
        self.v = v

    def update(self):
        pass


class LinearStep():
    def __init__(self, max, min, steps):
        self.steps = float(steps)
        self.max = max
        self.min = min
        self.cur_step = 0
        self.v = self.max

    def update(self):
        v = max(self.max - (self.max - self.min) *
                self.cur_step / self.steps, self.min)
        self.cur_step += 1
        self.v = v


class fc_block(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn):
        super(fc_block, self).__init__()

        block = nn.Sequential()
        block.add_module('linear', nn.Linear(in_channels, out_channels))
        if norm:
            block.add_module('batchnorm', nn.BatchNorm1d(out_channels))
        if activation_fn is not None:
            block.add_module('activation', activation_fn())

        self.block = block

    def forward(self, x):
        return self.block(x)


class conv_block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            norm,
            activation_fn):
        super(conv_block, self).__init__()

        block = nn.Sequential()
        block.add_module(
            'conv',
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride))
        if norm:
            block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            block.add_module('activation', activation_fn())

        self.block = block

    def forward(self, x):
        return self.block(x)


def get_conv_output_shape(shape, block):
    B = 1
    input = torch.rand(B, *shape)
    output = block(input)
    n_size = output.data.view(B, -1).size(1)
    return n_size


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def BHWC_to_BCHW(tensor):
    tensor = torch.transpose(tensor, 1, 3)  # BCWH
    tensor = torch.transpose(tensor, 2, 3)  # BCHW
    return tensor


def LCS(X, Y):

    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]
    longest_L = [[[]] * (n + 1) for i in range(m + 1)]
    longest = 0
    lcs_set = []

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
                longest_L[i][j] = []
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
                longest_L[i][j] = longest_L[i - 1][j - 1] + [X[i - 1]]
                if L[i][j] > longest:
                    lcs_set = []
                    lcs_set.append(longest_L[i][j])
                    longest = L[i][j]
                elif L[i][j] == longest and longest != 0:
                    lcs_set.append(longest_L[i][j])
            else:
                if L[i - 1][j] > L[i][j - 1]:
                    L[i][j] = L[i - 1][j]
                    longest_L[i][j] = longest_L[i - 1][j]
                else:
                    L[i][j] = L[i][j - 1]
                    longest_L[i][j] = longest_L[i][j - 1]

    if len(lcs_set) > 0:
        return lcs_set[0]
    else:
        return lcs_set
