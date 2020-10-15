import numpy as np
import torch
import torch.nn as nn
import os

import torch.nn.functional as F
import torchvision.models as models

from .distributions import Bernoulli, Categorical, DiagGaussian, ElementWiseCategorical


from utils.utils_models import init
from utils import utils_rl_agent

import ipdb
import pdb
import sys
from . import base_nets

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCritic(nn.Module):
    def __init__(self, action_space, base_name, base_kwargs=None, seed=0):
        self.rndnp = np.random.RandomState(seed)
        torch.manual_seed(seed)


        super(ActorCritic, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))


        context_type = 'avg'
        if base_name == 'TF':
            base = base_nets.TransformerBase
        elif base_name == 'GNN':
            base = base_nets.GraphEncoder
            #context_type = 'first'
        else:
            raise NotImplementedError

        node_encoder = base(**base_kwargs)
        self.hidden_size = base_kwargs['hidden_size']
        self.base = base_nets.GoalAttentionModel(hidden_size=base_kwargs['hidden_size'],
                                                 recurrent=True,
                                                 num_classes=base_kwargs['num_classes'],
                                                 node_encoder=node_encoder,
                                                 context_type=context_type)
        self.critic_linear = init_(nn.Linear(base_kwargs['hidden_size'], 1))

        # Distribution for the actions: Action, Obj1, Obj2
        dist = []
        for it, action_space_type in enumerate(action_space):
            num_outputs = action_space_type.n
            dist.append(Categorical(self.base.output_size, num_outputs))

        self.dist = nn.ModuleList(dist)

        # auxiliary nets
        # num_classes = base_kwargs['num_classes']
        # self.pred_close_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                                    nn.ReLU(),
        #                                    nn.Linear(self.hidden_size, 1))
        # self.pred_goal_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                                    nn.ReLU(),
        #                                    nn.Linear(self.hidden_size, num_classes))

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size


    def is_cuda(self):
        return torch.cuda.is_available()

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks=None, deterministic=False, epsilon=0.0, action_indices=None):

        if self.is_cuda():
            new_inputs = {}
            for name, inp in inputs.items():
                new_inputs[name] = inp.cuda()
            inputs = new_inputs
        affordance_obj1 = inputs['affordance_matrix']

        # value function, history, node_embedding, rnn
        context_goal, object_goal, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        value = self.critic_linear(context_goal)

        # TODO: this can probably be always shared across a batch
        object_classes = inputs['class_objects']
        mask_actions_nodes = inputs['mask_action_node']

        # select object1, and mask action accordingly
        indices = [0, 1] 

        actions = [None] * len(indices)
        actions_probs = [None] * len(indices)
        for i in indices:
            distr = self.dist[i]
            dist = distr(context_goal)
            log_probs = dist.original_logits

            # if i == 0:
            #   print(new_log_probs)
            # if i == 1:
            # if i == 1:
            #     print(new_log_probs)
            # Correct probabilities according to previously selected acitons
            if action_indices is None:
                u = self.rndnp.random()
                if u < epsilon:
                    uniform_logits = torch.ones(dist.original_logits.shape).to(log_probs.device)

                    random_policy = torch.distributions.Categorical(logits=uniform_logits)
                    action = random_policy.sample().unsqueeze(-1)

                else:
                    if deterministic:
                        action = dist.mode()
                    else:
                        action = dist.sample()
            else:
                action = action_indices[i].long()

            actions[i] = action
            actions_probs[i] = dist.probs



            #pdb.set_trace()
            #print('PROBABILITY', actions_probs[action])
        outputs = None
        return value, actions, actions_probs, rnn_hxs, outputs






