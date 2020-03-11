import torch
from models import actor_critic
from gym import spaces
from utils import utils_rl_agent
import numpy as np
import pdb
import copy

class RL_agent:
    """
    MCTS for a single agent
    """
    def __init__(self, args, agent_id, char_index, graph_helper, deterministic=False):

        self.agent_type = 'RL'
        self.max_num_objects = args.max_num_objects
        self.num_actions = graph_helper.num_actions
        self.num_object_classes = graph_helper.num_objects

        # TODO: encode states
        base_kwargs = {
            'hidden_size': args.hidden_size,
            'recurrent': True,
            'max_nodes': self.max_num_objects,
            'num_classes': self.num_object_classes

        }

        self.graph_helper = graph_helper

        self.agent_id = agent_id
        self.char_index = char_index

        self.epsilon = args.epsilon
        self.deterministic = deterministic

        self.hidden_size = args.hidden_size

        self.action_space = spaces.Tuple((spaces.Discrete(self.num_actions), spaces.Discrete(self.max_num_objects)))
        self.actor_critic = actor_critic.ActorCritic(self.action_space, base_name=args.base_net, base_kwargs=base_kwargs)

        self.id2node = None
        self.hidden_state = self.init_hidden_state()

    def init_hidden_state(self):
        h_state = torch.zeros(1, self.hidden_size)
        return h_state

    def reset(self, graph):

        self.id2node = {node['id']: node for node in graph['nodes']}
        self.hidden_state = self.init_hidden_state()


    def evaluate(self, rollout):
        pass

    def get_action(self, observation, goal_spec, action_indices=None):
        rnn_hxs = self.hidden_state

        inputs, info = self.graph_helper.build_graph(observation, character_id=self.agent_id)
        #assert(inputs.size(1) == 1)

        target_obj_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        target_loc_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        pre_id = 0
        for predicate, count in goal_spec.items():
            if count == 0:
                continue

            if not (predicate.startswith('on') or predicate.startswith('inside')):
                continue

            elements = predicate.split('_')
            obj_class_id = int(self.graph_helper.object_dict.get_id(elements[1]))
            loc_class_id = int(self.graph_helper.object_dict.get_id(self.id2node[int(elements[2])]['class_name']))
            for _ in range(count):
                target_obj_class[pre_id] = obj_class_id
                target_loc_class[pre_id] = loc_class_id
                pre_id += 1

        inputs.update({
            'affordance_matrix': self.graph_helper.obj1_affordance,
            'target_obj_class': target_obj_class,
            'target_loc_class': target_loc_class
        })

        inputs_tensor = {}
        for input_name, inp in inputs.items():
            inp_tensor = torch.tensor(inp).unsqueeze(0)
            if inp_tensor.type() == 'torch.DoubleTensor':
                inp_tensor = inp_tensor.float()
            inputs_tensor[input_name] = inp_tensor


        masks = torch.ones(rnn_hxs.shape).type(rnn_hxs.type())
        value, action, action_log_probs, rnn_state = self.actor_critic.act(inputs_tensor,
                                                                           rnn_hxs,
                                                                           masks,
                                                                           deterministic=self.deterministic,
                                                                           epsilon=self.epsilon,
                                                                           action_indices=action_indices)
        self.hidden_state = rnn_state
        info_model = {}
        info_model['log_probs'] = action_log_probs
        info_model['value'] = value
        info_model['actions'] = action
        info_model['state_inputs'] = copy.deepcopy(inputs)


        visible_objects = info[-1]

        action_str = self.get_action_instr(action, visible_objects, observation)
        return action_str, info_model


    def get_action_instr(self, action, visible_objects, current_graph):
        action_name = self.graph_helper.action_dict.get_el(action[0].item())
        object_id = action[1].item()

        (o1, o1_id) = visible_objects[object_id]
        if o1 == 'no_obj':
            o1 = None
        action = utils_rl_agent.can_perform_action(action_name, o1, o1_id, self.agent_id, current_graph)
        return action