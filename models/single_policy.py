import torch
import pdb
from torch import distributions, nn
import utils
from . import networks

class GoalEmbed(torch.nn.Module):
    def __init__(self, num_goals, out_dim):
        super(GoalEmbed, self).__init__()
        self.num_goals = num_goals
        self.out_dim = out_dim
        self.type_dim = 3
        self.goal_type_embed = nn.Embedding(self.num_goals, self.type_dim)
        self.goal_type_node_embed = nn.Linear(self.out_dim+self.type_dim, self.out_dim)

    def forward(self, goals, node_repr):
        goal_id, goal_classnode_id, goal_node_id = goals
        goal_type = self.goal_type_embed(goal_id)
        bs, tstps = node_repr.shape[:2]
        goal_type = goal_type.unsqueeze(-2).repeat(1, tstps, 1)
        node_goal = node_repr[torch.arange(bs), :, goal_node_id, :]
        goal_and_node = torch.cat([goal_type, node_goal], dim=2)
        return self.goal_type_node_embed(goal_and_node)

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

        self.fc_action = nn.Sequential(torch.nn.Linear(self.repr_dim * 2, helper.args.object_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(self.repr_dim, num_actions))

        self.fc_o1 = nn.Linear(self.repr_dim*3, 1)
        self.fc_o2 = nn.Linear(self.repr_dim*3, 1)
        # self.state_embedding = networks.StateRepresentation(helper, dataset)
        self.state_embedding = networks.GraphStateRepresentation(helper, dataset)

        # num_goals = num_objects
        num_goals = len(dataset.object_dict)
        self.goal_embedding = GoalEmbed(num_goals, self.repr_dim)
        self.goal_embedding.cuda()



        # Combine char and object selected
        self.objectchar = nn.Sequential(torch.nn.Linear(helper.args.object_dim*2, helper.args.object_dim),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(helper.args.object_dim, helper.args.object_dim))

        self.objectactionchar = nn.Sequential(torch.nn.Linear(helper.args.object_dim * 3,
                                                              helper.args.object_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(helper.args.object_dim, helper.args.object_dim))

        # Debug:
        self.interm_grad = None

        def debug_var(x):
            print('Backward')
            self.interm_grad = x
        self.debug_var = debug_var

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

    def forward(self, observations, goals):

        # Obtain the initial node representations
        num_steps = observations[0].shape[1]
        class_names, class_ids, states, edges, edge_types, visibility, mask_nodes, mask_edges = observations
        node_repr, global_repr = self.state_embedding(observations)
        goal_repr = self.goal_embedding(goals, node_repr)

        num_nodes = node_repr.shape[-2]

        concat_nodes = torch.cat([node_repr, global_repr.unsqueeze(-2).repeat(1, 1, num_nodes, 1),
                                             goal_repr.unsqueeze(-2).repeat(1, 1, num_nodes, 1)], -1)
        global_and_goal = torch.cat([goal_repr, global_repr], 2)
        action_logits = self.fc_action(global_and_goal)
        node_1_logits = self.fc_o1(concat_nodes).squeeze(-1)
        node_2_logits = self.fc_o2(concat_nodes).squeeze(-1)

        # Mask out
        node_1_logits = node_1_logits * visibility + (1-visibility) * -1e6
        node_2_logits = node_2_logits * visibility + (1-visibility) * -1e6
        # Predict actions according to the global representation
        # pdb.set_trace()
        return action_logits, node_1_logits, node_2_logits, (node_repr, global_repr)



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

