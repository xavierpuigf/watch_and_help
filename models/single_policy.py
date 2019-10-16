import torch
import pdb
from torch import distributions, nn
import utils


class SinglePolicy(torch.nn.Module):
    class SinglePolicyActivations:
        # This class will store for every agent it's own activations
        def __init__(self, helper):
            self.action_embedding = None
            self.object_embedding = None

    def __init__(self, dataset, helper):
        super(SinglePolicy, self).__init__()

        # Network
        self.helper = helper
        self.dataset = dataset
        self.action_embedding = nn.Embedding(len(dataset.action_dict), helper.args.action_dim)
        self.object_embedding = nn.Embedding(len(dataset.object_dict), helper.args.object_dim)
        self.state_embedding = nn.Embedding(len(dataset.state_dict), helper.args.state_dim)
        self.relation_embedding = nn.Embedding(len(dataset.relation_dict), helper.args.relation_dim)

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
        # Returns a distribution of actions based on the policy
        action_ids = torch.tensor([self.dataset.action_dict.get_id(action) for action in actions])
        action_embeddings = self.action_embedding(action_ids)
        embed_character = self.object_embedding(torch.tensor(self.dataset.object_dict.get_id('character')))
        #print(agent.policy_net)
        embed_character_o1 = self.objectchar(
            torch.cat([embed_character, agent.activation_info.object_embedding[object_selected]], 0))[None, :]
        agent.activation_info.action_embedding = action_embeddings
        logits = (action_embeddings*embed_character_o1).sum(1)
        distr = distributions.categorical.Categorical(logits=logits)

        return distr, actions

    def get_first_obj(self, observations, agent): # Inside the environment
        # Obtain the node names
        candidates = observations['nodes'] + [None]
        node_names = [node['class_name'] if type(node) == dict else None for node in candidates]
        node_names = [x for x in node_names if x != 'character']

        node_name_ids = torch.tensor(
            [self.dataset.object_dict.get_id(node_name) for node_name in node_names]).to(dtype=torch.long)
        node_embeddings = self.object_embedding(node_name_ids)
        agent.activation_info.object_embedding = node_embeddings

        node_character = self.object_embedding(torch.tensor(self.dataset.object_dict.get_id('character')))
        logit_attention = (node_embeddings*node_character[None, :]).sum(1)
        distr = distributions.categorical.Categorical(logits=logit_attention)
        return distr, candidates

    def get_second_obj(self, triples, agent, object_selected, action_selected): # Inside the environment

        candidates = triples['nodes'] + [None]
        node_names = [node['class_name'] if type(node) == dict else None for node in candidates]
        node_name_ids = torch.tensor([self.dataset.object_dict.get_id(node_name) for node_name in node_names])
        node_embeddings = self.object_embedding(node_name_ids)
        pdb.set_trace()
        self.object_embedding = node_embeddings

        node_character = self.object_embedding(self.dataset.object_dict.get_id('character'))
        logit_attention = (node_embeddings * node_character[None, :]).sum(1)
        distr = distributions.categorical.Categorical(logits=logit_attention)
        return distr, candidates

    def forward(self, observations):
        return []

    def bc_loss(self, program, agent_info):
        # Computes crossentropy loss between actions taken and gt actions

        action_space = agent_info['action_space']
        saved_log_probs = agent_info['saved_log_probs']
        num_steps = len(action_space)
        actions, o1, o2 = utils.parse_prog(program)
        action_candidates = [x[1] for x in action_space]
        obj1_candidates = [[(node['class_name'], node['id']) if node is not None else None for node in x[0]] for x in action_space]
        if len(action_space[0]) > 2:
            # Assumption here is that all the actions will have the same #args
            # not necessarily true though
            obj2_candidates = [[(node['class_name'], node['id']) for node in x[1]] for x in action_space]
        else:
            obj2_candidates = [[None] for _ in range(num_steps)]

        # Obtain the candidates and find the matching index
        # Loss o1
        losses, losses_object1, losses_object2, losses_action = [], [], [], []
        for it in range(num_steps):
            loss = torch.zeros([1])
            loss_object1 = torch.zeros([1])
            loss_object2 = torch.zeros([1])
            loss_action = torch.zeros([1])
            gt_action, gt_o1, gt_o2 = actions[it], o1[it], o2[it]
            index_action = [it for it,x in enumerate(action_candidates[it]) if x.upper() == gt_action]
            index_o1 = [it for it, x in enumerate(obj1_candidates[it]) if x == gt_o1]
            index_o2 = [it for it, x in enumerate(obj2_candidates[it]) if x == gt_o2]
            index_action = index_action[0] if len(index_action) > 0 else None
            index_o1 = index_o1[0] if len(index_o1) > 0 else None
            index_o2 = index_o2[0] if len(index_o2) > 0 else None
            if len(action_candidates[it]) > 1 and index_action is not None:
                loss += -saved_log_probs[it][1] # action
                loss_action += -saved_log_probs[it][1]
                #print('action_loss', -saved_log_probs[it][1])
            if len(obj1_candidates[it]) > 1 and index_o1 is not None:
                loss += -saved_log_probs[it][0] # object1
                loss_object1 += -saved_log_probs[it][0]
                print(loss_object1)
            else:
                if it == 0:
                    pdb.set_trace()
                #print('o1', -saved_log_probs[it][0])
            if len(obj2_candidates[it]) > 1 and index_o2 is not None:
                loss += -saved_log_probs[it][2] # object2
                loss_object2 +=  -saved_log_probs[it][2]
                #print('o2')
            losses.append(loss)
            losses_object1.append(loss_object1)
            losses_object2.append(loss_object2)
            losses_action.append(loss_action)
        #print(losses)
        losses = torch.cat(losses)
        losses_o1 = torch.cat(losses_object1)
        losses_o2 = torch.cat(losses_object2)
        losses_action = torch.cat(losses_action)

        return losses.mean(), losses_action.mean(), losses_o1.mean(), losses_o2.mean()

    def pg_loss(self, labels, agent_info):
        rewards = agent_info['rewards']
        saved_log_probs = agent_info['saved_log_probs']
        num_step = len(rewards)
        policy_loss = []
        for it in num_step:
            reward = rewards[it]
            log_prob = saved_log_probs[it]
            policy_loss.append(-log_prob*reward)
        return torch.cat(policy_loss).sum()

