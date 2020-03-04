import numpy as np
import torch
import torch.nn as nn
import os

import torch.nn.functional as F
import torchvision.models as models

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, ElementWiseCategorical
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.graph_nn import Transformer, GraphModel
import pdb
import  sys
home_path = os.getcwd()
home_path = '/'.join(home_path.split('/')[:-2])
sys.path.append(home_path+'/vh_multiagent_models')
import utils_rl_agent

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_space, action_space, action_inst=True, attention_type='dot', base=None, base_kwargs=None):
        super(Policy, self).__init__()

        self.attention_type = attention_type


        self.base = base(**base_kwargs)
        if action_space.__class__.__name__ != "Tuple":
            action_space = [action_space]
        
        dist = []
        for it, action_space_type in enumerate(action_space):
            if action_space_type.__class__.__name__ == "Discrete":
                num_outputs = action_space_type.n
                if action_inst and it > 0:
                    if self.attention_type == 'linear':
                        dist.append(ElementWiseCategorical(self.base.output_size,
                                                           method=attention_type))
                    else:
                        dist.append(ElementWiseCategorical(self.base.output_size+self.base.context_size, method=attention_type))
                else:
                    dist.append(Categorical(self.base.output_size, num_outputs))
            elif action_space_type.__class__.__name__ == "Box":
                num_outputs = action_space_type.shape[0]
                dist.append(DiagGaussian(self.base.output_size, num_outputs))
            elif action_space_type.__class__.__name__ == "MultiBinary":
                num_outputs = action_space_type.shape[0]
                dist.append(Bernoulli(self.base.output_size, num_outputs))
            else:
                raise NotImplementedError
        self.dist = nn.ModuleList(dist)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, epsilon=0.0, action_indices=None):

        affordance_obj1 = inputs['affordance_matrix']


        # value function, history, node_embedding, rnn
        outputs = self.base(inputs, rnn_hxs, masks)
        object_classes = inputs['class_objects']

        # if object_classes.shape[0] != 1:
        #     pdb.set_trace()
        mask_observations = inputs['mask_object']

        if len(outputs) == 3:
            value, actor_features, rnn_hxs = outputs
            summary_nodes = actor_features
        else:
            value, summary_nodes, actor_features, rnn_hxs = outputs

        # select object1, and mask action accordingly
        if len(self.dist) > 1:
            indices = [1, 0] # object1, action
        else:
            indices = list(range(len(self.dist)))

        actions = [None] * len(indices)
        actions_log_probs = [None] * len(indices)
        for i in indices:
            distr = self.dist[i]
            if i == 0:
                dist = distr(summary_nodes)
            else:
                dist = distr(summary_nodes, actor_features)

            new_log_probs = utils_rl_agent.update_probs(dist.original_logits, i, actions, object_classes, mask_observations, affordance_obj1)
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
                                                                         mask_observations, affordance_obj1)


                    random_policy = torch.distributions.Categorical(logits=new_log_probs)
                    action = random_policy.sample().unsqueeze(-1)

                else:
                    if deterministic:
                        action = dist.mode()
                    else:
                        try:
                            action = dist.sample()
                        except:
                            pdb.set_trace()
            else:
                action = action_indices[i][0].long()

            actions[i] = action
            # print(new_log_probs.shape)
            actions_log_probs[i] = dist.log_probs(action)

            dist_entropy = dist.entropy().mean()
        return value, actions, actions_log_probs, rnn_hxs

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


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class GraphEncoder(nn.Module):
    def __init__(self, hidden_size=512, num_nodes=100, num_rels=5, num_classes=150):

        super(GraphEncoder, self).__init__()
        self.hidden_size=hidden_size
        self.graph_encoder = GraphModel(
                num_classes=num_classes,
                num_nodes=num_nodes, h_dim=hidden_size, out_dim=hidden_size, num_rels=num_rels, max_nodes=num_nodes)

    def forward(self, inputs):
        # Build the graph
        hidden_feats = self.graph_encoder(inputs)
        return hidden_feats


class TransformerBase(NNBase):

    def __init__(self, recurrent=False, hidden_size=128, dist_size=10, max_nodes=150, num_classes=100):
        super(TransformerBase, self).__init__(recurrent, hidden_size, hidden_size)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.main = Transformer(num_classes=num_classes, num_nodes=max_nodes, in_feat=hidden_size, out_feat=hidden_size)
        self.context_size = hidden_size
        self.critic_linear = init_(nn.Linear(hidden_size, 1))


        self.object_context_combine = self.mlp2l(2*hidden_size, hidden_size)
        self.single_object_encoding = ObjNameCoordEncode(output_dim=hidden_size, num_classes=num_classes)
        self.goal_encoder = nn.EmbeddingBag(num_classes, hidden_size, mode='sum')

        self.fc_att_action = self.mlp2l(hidden_size * 2, hidden_size)
        self.fc_att_object = self.mlp2l(hidden_size * 2, hidden_size)
        self.train()

    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out))


    def forward(self, inputs, rnn_hxs, masks):

        # Use transformer to get feats for every object
        mask_visible = inputs['mask_object']
        input_node_embedding = self.single_object_encoding(inputs['class_objects'].long(), inputs['object_coords']).squeeze(1)
        features_obj = self.main(input_node_embedding, mask_visible)

        # 1 x ndim. Avg pool the features for the context vec
        mask_visible = mask_visible.unsqueeze(-1)

        # Mean pool of transformer
        context_vec = (features_obj * mask_visible).sum(1) / (1e-9+mask_visible.sum(1))


        # Goal embedding
        obj_class_name = inputs['target_obj_class'][:, 0].long()
        loc_class_name = inputs['target_loc_class'][:, 0].long()
        goal_encoding_obj = self.goal_encoder(obj_class_name).squeeze(1)
        goal_encoding_loc = self.goal_encoder(loc_class_name).squeeze(1)
        goal_encoding = torch.cat([goal_encoding_obj, goal_encoding_loc], dim=-1)
        goal_mask_action = torch.sigmoid(self.fc_att_action(goal_encoding))
        goal_mask_object = torch.sigmoid(self.fc_att_object(goal_encoding))

        # Recurrent context
        # if self.is_recurrent:
        r_context_vec, rnn_hxs = self._forward_gru(context_vec, rnn_hxs, masks)

        # h' = GA . h [bs, h]
        context_goal = goal_mask_action * r_context_vec

        # Combine object representations with global representations
        r_object_vec = torch.cat([features_obj, r_context_vec.unsqueeze(1).repeat(1, features_obj.shape[1], 1)], 2)
        r_object_vec_comb = self.object_context_combine(r_object_vec)

        # Sg' = GA . Sg [bs, N, h]
        object_goal = goal_mask_object[:, None, :] * r_object_vec_comb

        if torch.isnan(context_goal).any() or torch.isnan(object_goal).any():
            pdb.set_trace()

        return self.critic_linear(context_goal), context_goal, object_goal, rnn_hxs




class GraphBase(NNBase):
    def __init__(self, recurrent=False, hidden_size=128, max_nodes=150, num_classes=50):
        super(GraphBase, self).__init__(recurrent, hidden_size, hidden_size)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = GraphEncoder(hidden_size, num_nodes=max_nodes, num_classes=num_classes)
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):

        if inputs[0].ndim > 3:
            # first element is an image
            # x: [bs, num_objects, dim] (x[:, 0, :] always character)
            x = self.main(inputs)
        else:
            x = self.main(inputs)

        char_node = x[:, 0]
        if self.is_recurrent:
            char_node, rnn_hxs = self._forward_gru(char_node, rnn_hxs, masks)
        return self.critic_linear(char_node), char_node, x, rnn_hxs


class CNNBaseResnetDist(NNBase):
    def __init__(self, recurrent=False, hidden_size=512, dist_size=10):
        super(CNNBaseResnetDist, self).__init__(recurrent, hidden_size, hidden_size)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = torch.nn.Sequential(
                *(list(models.resnet50(pretrained=True).children())[:-1]),
                nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.ReLU(),
                Flatten())
                
        self.base_dist = torch.nn.Sequential(nn.Linear(2, dist_size), nn.ReLU()) 
        self.combine = nn.Sequential(
                init_(nn.Linear(hidden_size+dist_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs['image'])
        y = self.base_dist(inputs['rel_dist'])
        x = self.combine(torch.cat([x, y], dim=1))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs


class ObjNameCoordEncode(nn.Module):
    def __init__(self, output_dim=128, num_classes=50):
        super(ObjNameCoordEncode, self).__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(num_classes, int(output_dim/2))
        self.coord_embedding = nn.Sequential(nn.Linear(3, int(output_dim/2)),
                                             nn.ReLU(),
                                             nn.Linear(int(output_dim/2), int(output_dim/2)))

    def forward(self, class_ids, coords):
        class_embedding = self.class_embedding(class_ids)
        coord_embedding = self.coord_embedding(coords)
        return torch.cat([class_embedding, coord_embedding], dim=2)


class CNNBaseResnet(NNBase):
    def __init__(self, recurrent=False, hidden_size=128, num_classes=50):
        super(CNNBaseResnet, self).__init__(recurrent, hidden_size, hidden_size)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = torch.nn.Sequential(
                *(list(models.resnet50(pretrained=True).children())[:-1]),
                nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False),
                Flatten())

        self.context_size = 10
        self.class_embedding = nn.Embedding(num_classes, self.context_size)



        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()


    def forward(self, inputs, rnn_hxs, masks):

        x = self.main(inputs['image'])

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        context = self.class_embedding(inputs['class_objects'].long())
        return self.critic_linear(x), x, context, rnn_hxs

class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 10 * 10, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
