import numpy as np
import torch
import torch.nn as nn
import os
import ipdb

import torch.nn.functional as F
import torchvision.models as models

from .distributions import Bernoulli, Categorical, DiagGaussian, ElementWiseCategorical


from utils.utils_models import init
from utils import utils_rl_agent

import pdb
import sys
from . import base_nets







class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCritic(nn.Module):
    def __init__(self, action_space, base_name, base_kwargs=None, seed=0):
        np.random.seed(seed)
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

        # Distribution for the actions
        dist = []
        for it, action_space_type in enumerate(action_space):
            num_outputs = action_space_type.n
            # first distribution selects over instances, second over actions
            if it > 0:
                dist.append(ElementWiseCategorical(self.base.output_size, method='linear'))
            else:
                dist.append(Categorical(self.base.output_size, num_outputs))

        self.dist = nn.ModuleList(dist)

        # auxiliary nets
        num_classes = base_kwargs['num_classes']
        self.pred_close_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, 1))
        self.pred_goal_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, num_classes))

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
        # ipdb.set_trace()
        if self.is_cuda():
            new_inputs = {}
            for name, inp in inputs.items():
                new_inputs[name] = inp.cuda()
            inputs = new_inputs
        affordance_obj1 = inputs['affordance_matrix']
        # ipdb.set_trace()
        # value function, history, node_embedding, rnn
        context_goal, object_goal, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        value = self.critic_linear(context_goal)

        # TODO: this can probably be always shared across a batch
        object_classes = inputs['class_objects']
        mask_actions_nodes = inputs['mask_action_node']

        # select object1, and mask action accordingly
        indices = [1, 0] # object1, action

        actions = [None] * len(indices)
        actions_probs = [None] * len(indices)
        for i in indices:
            distr = self.dist[i]
            if i == 0:
                dist = distr(context_goal)
            else:
                dist = distr(context_goal, object_goal)

            new_log_probs = utils_rl_agent.update_probs(dist.original_logits, i, actions, object_classes, mask_actions_nodes, affordance_obj1)
            # if i == 0:
            #   print(new_log_probs)
            dist = distr.update_logs(new_log_probs)
            # if i == 1:
            # if i == 1:
            #     print(new_log_probs)
            # Correct probabilities according to previously selected acitons
            if action_indices is None:
                u = np.random.random()
                if u < epsilon:
                    uniform_logits = torch.ones(dist.original_logits.shape).to(new_log_probs.device)
                    updated_uniform_logits = utils_rl_agent.update_probs(uniform_logits, i, actions, object_classes,
                                                                         mask_actions_nodes, affordance_obj1)

                    random_policy = torch.distributions.Categorical(logits=updated_uniform_logits)
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
        outputs = {
            'context_goal': context_goal,
            'object_goal': object_goal
        }
        return value, actions, actions_probs, rnn_hxs, outputs


    def auxiliary_pred(self, inputs):
        pred_close = self.pred_close_net(inputs['object_goal'])
        pred_goal = self.pred_goal_net(inputs['context_goal'])

        outputs = {
            'pred_goal': pred_goal,
            'pred_close': pred_close
        }
        return outputs

    def get_value(self, inputs, rnn_hxs, masks):

        outputs_model = self.base(inputs, rnn_hxs, masks)
        return outputs_model[0]

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        outputs_model = self.base(inputs, rnn_hxs, masks)
        if len(outputs_model) == 3:
            value, actor_features, rnn_hxs = outputs_model
            summary_nodes = actor_features
        else:
            value, summary_nodes, actor_features, rnn_hxs = outputs_model
        
        action_log_probs = []
        dist_entropy = []
        for i, distr in enumerate(self.dist):
            if i == 0:
                dist = distr(summary_nodes)
            else:
                dist = distr(summary_nodes, actor_features)

            action_log_probs.append(dist.log_probs(action[i]))
            dist_entropy.append(dist.entropy().mean())

        return value, action_log_probs, dist_entropy, rnn_hxs

