import torch
import pdb
from torch import distributions, nn
import utils
from . import networks


class SinglePolicy(torch.nn.Module):
    class SinglePolicyActivations:
        # This class will store for every agent it's own activations
        def __init__(self, helper):
            # The previous action used
            self.action_embedding = None
            self.object_embedding = None
            self.state = None

    def __init__(self, dataset, helper):
        super(SinglePolicy, self).__init__()

        # Network
        self.node_None = {'id': -1, 'class_name': None, 'properties': [], 'states': []}
        self.node_Stop = {'id': -2, 'class_name': 'stop', 'properties': [], 'states': []}

        self.helper = helper
        self.dataset = dataset
        self.repr_dim = 100

        num_actions = len(dataset.action_dict)

        self.fc_action = nn.Sequential(torch.nn.Linear(self.repr_dim, helper.args.object_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(self.repr_dim, num_actions))

        self.fc_o1 = nn.Linear(self.repr_dim*2, 1)
        self.fc_o2 = nn.Linear(self.repr_dim*2, 1)
        # self.state_embedding = networks.StateRepresentation(helper, dataset)
        self.state_embedding = networks.GraphStateRepresentation(helper, dataset)

        # Combine char and object selected
        self.objectchar = nn.Sequential(torch.nn.Linear(helper.args.object_dim*2, helper.args.object_dim),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(helper.args.object_dim, helper.args.object_dim))

        self.objectactionchar = nn.Sequential(torch.nn.Linear(helper.args.object_dim * 3,
                                                              helper.args.object_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(helper.args.object_dim, helper.args.object_dim))

    def activation_info(self):
        return self.SinglePolicyActivations(self.helper)

    def get_action(self, actions, agent, object_selected):
        is_cuda = next(self.parameters()).is_cuda
        # Returns a distribution of actions based on the policy
        action_ids = torch.tensor([self.dataset.action_dict.get_id(action) for action in actions])
        if is_cuda:
            action_ids = action_ids.cuda()
        action_embeddings = self.action_embedding(action_ids)

        embed_character = agent.activation_info.state['char_embedding']
        embed_character_o1 = self.objectchar(
            torch.cat([embed_character,
                       agent.activation_info.state['node_embeddings'][object_selected][None, :]], 1))
        agent.activation_info.action_embedding = action_embeddings

        logits = (action_embeddings*embed_character_o1).sum(1)

        return logits, actions

    def get_first_obj(self, observations, agent): # Inside the environment
        is_cuda = next(self.parameters()).is_cuda


        # Obtain the node names
        # TODO: We should include the node None as well
        observations['nodes'] = observations['nodes'] + [self.node_Stop]

        node_embeddings, char_embedding = self.state_embedding(observations)

        agent.activation_info.state = {'node_embeddings': node_embeddings,
                                       'char_embedding': char_embedding
                                      }
        # Remove character
        candidates = [x for it, x in enumerate(observations['nodes']) if x['class_name'] != 'character']
        index_candidates = [it for it, x in enumerate(observations['nodes']) if x['class_name'] != 'character']
        index_candidates = torch.tensor(index_candidates)

        logit_attention = (node_embeddings*char_embedding).sum(1)
        logit_attention = logit_attention[index_candidates]

        # print(node_embeddings.shape, char_embedding.shape, logit_attention.shape)

        return logit_attention, candidates

    def get_second_obj(self, triples, agent, object_selected, action_selected): # Inside the environment
        # TODO: finish implementing
        pass
        # is_cuda = next(self.parameters()).is_cuda
        # candidates = triples['nodes'] + [None]
        # node_names = [node['class_name'] if type(node) == dict else node for node in candidates]
        # node_name_ids = torch.tensor([self.dataset.object_dict.get_id(node_name) for node_name in node_names])
        # node_embeddings = self.object_embedding(node_name_ids)
        # pdb.set_trace()
        # self.object_embedding = node_embeddings
        #
        # node_character = self.object_embedding(self.dataset.object_dict.get_id('character'))
        # logit_attention = (node_embeddings * node_character[None, :]).sum(1)
        # #distr = distributions.categorical.Categorical(logits=logit_attention)
        # return logit_attention, candidates

    def forward(self, observations):

        # Obtain the initial node representations
        class_names, class_ids, states, edges, edge_types, visibility, mask_edges = observations
        node_repr, global_repr = self.state_embedding(observations)
        num_nodes = node_repr.shape[-2]

        concat_nodes = torch.cat([node_repr, global_repr.unsqueeze(-2).repeat(1, 1, num_nodes, 1)], -1)
        action_logits = self.fc_action(global_repr)
        node_1_logits = self.fc_o1(concat_nodes).squeeze(-1)
        node_2_logits = self.fc_o2(concat_nodes).squeeze(-1)

        # Mask out
        node_1_logits = node_1_logits * visibility + (1-visibility) * -1e6
        node_2_logits = node_1_logits * visibility + (1-visibility) * -1e6
        # Predict actions according to the global representation

        return action_logits, node_1_logits, node_2_logits

    def bc_loss(self, program, agent_info):
        # Deprecated
        # Computes crossentropy loss between actions taken and gt actions
        action_space = agent_info['action_space']
        saved_log_probs = agent_info['saved_log_probs']
        is_cuda = saved_log_probs[0][0].is_cuda

        num_steps = min(len(program), len(action_space))

        # GT action
        actions, o1, o2 = utils.parse_prog(program)
        action_candidates = [x[1] if len(x) > 1 else ['None'] for x in action_space]
        obj1_candidates = [[(node['class_name'], node['id']) for node in x[0]] for x in action_space]
        obj2_candidates = [[(node['class_name'], node['id']) for node in x[2]] if len(x) > 2 else [None] for x in action_space]


        # Obtain the candidates and find the matching index
        # Loss o1
        losses, losses_object1, losses_object2, losses_action = [], [], [], []
        mlosses, mlosses_object1, mlosses_object2, mlosses_action = [], [], [], []
        for it in range(num_steps):
            loss = torch.zeros([1])
            loss_object1 = torch.zeros([1])
            loss_object2 = torch.zeros([1])
            loss_action = torch.zeros([1])

            if is_cuda:
                loss = loss.cuda()
                loss_object1 = loss_object1.cuda()
                loss_object2 = loss_object2.cuda()
                loss_action = loss_action.cuda()


            gt_action, gt_o1, gt_o2 = actions[it], o1[it], o2[it]
            index_action = [it for it,x in enumerate(action_candidates[it]) if x.upper() == gt_action]
            index_o1 = [it for it, x in enumerate(obj1_candidates[it]) if x == gt_o1]
            index_o2 = [it for it, x in enumerate(obj2_candidates[it]) if x == gt_o2]
            index_action = index_action[0] if len(index_action) > 0 else None
            index_o1 = index_o1[0] if len(index_o1) > 0 else None
            index_o2 = index_o2[0] if len(index_o2) > 0 else None

            # Hack, we always want to run action 2
            if gt_o2 is None:
                index_o2 = None

            valid_triple, valid_action, valid_o1, valid_o2 = False, False, False, False
            if index_action is not None:
                valid_triple = True
                loss += -saved_log_probs[it][1][index_action] # action
                loss_action += -saved_log_probs[it][1][index_action]
                valid_action = True

            if index_o1 is not None:
                valid_triple = True
                valid_o1 = True
                loss += -saved_log_probs[it][0][index_o1] # object1
                loss_object1 += -saved_log_probs[it][0][index_o1]

            if index_o2 is not None:
                valid_triple = True
                valid_o2 = True
                loss += -saved_log_probs[it][2][index_o2] # object2
                loss_object2 += -saved_log_probs[it][2][index_o2]

            losses.append(loss)
            losses_object1.append(loss_object1)
            losses_object2.append(loss_object2)
            losses_action.append(loss_action)
            mlosses.append(valid_triple)
            mlosses_object1.append(valid_o1)
            mlosses_object2.append(valid_o2)
            mlosses_action.append(valid_action)

        losses = torch.cat(losses)
        losses_o1 = torch.cat(losses_object1)
        losses_o2 = torch.cat(losses_object2)
        losses_action = torch.cat(losses_action)

        mlosses = torch.tensor(mlosses).float()
        mlosses_o1 = torch.tensor(mlosses_object1).float()
        mlosses_o2 = torch.tensor(mlosses_object2).float()
        mlosses_action = torch.tensor(mlosses_action).float()

        if is_cuda:
            mlosses = mlosses.cuda()
            mlosses_o1 = mlosses_o1.cuda()
            mlosses_o2 = mlosses_o2.cuda()
            mlosses_action = mlosses_action.cuda()


        mlosses /= (mlosses.sum()+1e-8)
        mlosses_o1 /= (mlosses_o1.sum()+1e-8)
        mlosses_o2 /= (mlosses_o2.sum()+1e-8)
        mlosses_action /= (mlosses_action.sum()+1e-8)

        loss = (losses * mlosses).sum()
        loss_action = (losses_action * mlosses_action).sum()
        loss_o1 = (losses_o1 * mlosses_o1).sum()
        loss_o2 = (losses_o2 * mlosses_o2).sum()

        return loss, loss_action, loss_o1, loss_o2

    def pg_loss(self, labels, agent_info):

        rewards = agent_info['rewards']
        saved_log_probs = agent_info['saved_log_probs']
        indices = agent_info['indices']
        num_step = len(rewards)
        policy_loss = []
        for it in num_step:
            reward = rewards[it]

            index = indices[it][0]
            log_prob = saved_log_probs[it][0][index]
            policy_loss.append(-log_prob*reward)
        return torch.cat(policy_loss).sum()

