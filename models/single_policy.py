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
        def __init__(self, dummy_var):
            # The previous action used
            self.action_embedding = None
            self.object_embedding = None
            self.state = None

    def __init__(self, dataset):
        super(SinglePolicy, self).__init__()
        self.clip_value = -1e9

        # Network
        self.node_None = {'id': -1, 'class_name': None, 'properties': [], 'states': []}
        self.node_Stop = {'id': -2, 'class_name': 'stop', 'properties': [], 'states': []}

        self.dataset = dataset
        self.repr_dim = self.dataset.args.state_dim

        num_actions = len(dataset.action_dict)

        self.fc_action = nn.Sequential(torch.nn.Linear(self.repr_dim * 2, dataset.args.object_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(self.repr_dim, num_actions))

        self.fc_o1 = nn.Sequential(nn.Linear(self.repr_dim*3, self.repr_dim), torch.nn.ReLU(), torch.nn.Linear(self.repr_dim, 1))
        self.fc_o2 = nn.Sequential(nn.Linear(self.repr_dim*3, self.repr_dim), torch.nn.ReLU(), torch.nn.Linear(self.repr_dim, 1))

        self.state_embedding = networks.GraphStateRepresentation(dataset)

        # num_goals = num_objects
        num_goals = len(dataset.object_dict)
        self.goal_embedding = GoalEmbed(num_goals, self.repr_dim)
        self.goal_embedding.cuda()



        # Combine char and object selected
        self.objectchar = nn.Sequential(torch.nn.Linear(dataset.args.object_dim*2, dataset.args.object_dim),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(dataset.args.object_dim, dataset.args.object_dim))

        self.objectactionchar = nn.Sequential(torch.nn.Linear(dataset.args.object_dim * 3,
                                                              dataset.args.object_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(dataset.args.object_dim, dataset.args.object_dim))

        # Debug:
        self.interm_grad = None

        def debug_var(x):
            print('Backward')
            self.interm_grad = x
        self.debug_var = debug_var

    def activation_info(self):
        # [Deprecated]
        return self.SinglePolicyActivations(self.dataset)



    def forward(self, observations, goals, class_id_char):

        # Obtain the initial node representations
        num_steps = observations[0].shape[1]
        class_names, class_ids, states, edges, edge_types, visibility, mask_nodes, mask_edges = observations
        node_repr, global_repr = self.state_embedding(observations)
        global_repr = ((class_names == class_id_char).unsqueeze(-1).float()*node_repr).sum(-2)
        goal_repr = self.goal_embedding(goals, node_repr)

        num_nodes = node_repr.shape[-2]

        concat_nodes = torch.cat([node_repr, global_repr.unsqueeze(-2).repeat(1, 1, num_nodes, 1),
                                             goal_repr.unsqueeze(-2).repeat(1, 1, num_nodes, 1)], -1)
        global_and_goal = torch.cat([goal_repr, global_repr], 2)
        action_logits = self.fc_action(global_and_goal)
        node_1_logits = self.fc_o1(concat_nodes).squeeze(-1)
        node_2_logits = self.fc_o2(concat_nodes).squeeze(-1)

        # Mask out
        action_logits = torch.clamp(action_logits, self.clip_value/2., -self.clip_value/2.)
        node_1_logits = torch.clamp(node_1_logits, self.clip_value/2., -self.clip_value/2.) * visibility + (1-visibility) * self.clip_value
        node_2_logits = torch.clamp(node_2_logits, self.clip_value/2., -self.clip_value/2.) * visibility + (1-visibility) * self.clip_value
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

