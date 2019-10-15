import torch

class SinglePolicy(torch.nn.Module):
    def __init__(self):
        super(SinglePolicy, self).__init__()

    def forward(self, observations):
        return []