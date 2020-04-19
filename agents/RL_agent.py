import torch
from models import actor_critic
from gym import spaces
from utils import utils_rl_agent
import numpy as np
import pdb
import copy
import random

class RL_agent:
    """
    RL for a single agent
    """
    def __init__(self, args, agent_id, char_index, graph_helper, deterministic=False, seed=123):
        self.args = args
        self.agent_type = 'RL'
        self.max_num_objects = args.max_num_objects
        self.num_actions = graph_helper.num_actions
        self.num_object_classes = graph_helper.num_classes
        self.num_states = graph_helper.num_states

        # TODO: encode states
        base_kwargs = {
            'hidden_size': args.hidden_size,
            'max_nodes': self.max_num_objects,
            'num_classes': self.num_object_classes,
            'num_states': self.num_states

        }

        self.graph_helper = graph_helper

        self.agent_id = agent_id
        self.char_index = char_index

        self.epsilon = args.init_epsilon
        self.deterministic = deterministic

        self.hidden_size = args.hidden_size

        self.action_space = spaces.Tuple((spaces.Discrete(self.num_actions), spaces.Discrete(self.max_num_objects)))
        self.actor_critic = actor_critic.ActorCritic(self.action_space, base_name=args.base_net,
                                                     base_kwargs=base_kwargs, seed=seed)

        self.id2node = None
        self.hidden_state = self.init_hidden_state()

        if torch.cuda.is_available():
            self.actor_critic.cuda()



    def init_hidden_state(self):
        h_state = torch.zeros(1, self.hidden_size)
        c_state = torch.zeros(1, self.hidden_size)
        return (h_state, c_state)

    def reset(self, graph):

        self.id2node = {node['id']: node for node in graph['nodes']}
        self.hidden_state = self.init_hidden_state()


    def evaluate(self, rollout):
        pass

    def get_action(self, observation, goal_spec, action_space_ids=None, action_indices=None):
        rnn_hxs = self.hidden_state

        masks = torch.ones(rnn_hxs[0].shape).type(rnn_hxs[0].type())
        if torch.cuda.is_available():
            rnn_hxs = (rnn_hxs[0].cuda(), rnn_hxs[1].cuda())
            masks = masks.cuda()
        inputs, info = self.graph_helper.build_graph(observation,
                                                     action_space_ids=action_space_ids,
                                                     character_id=self.agent_id)
        visible_objects = info[-1]

        target_obj_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        target_loc_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        mask_goal_pred = [0.0] * 6

        pre_id = 0
        for predicate, info in goal_spec.items():
            count, required, reward = info
            if count == 0 or not required:
                continue

            # if not (predicate.startswith('on') or predicate.startswith('inside')):
            #     continue

            elements = predicate.split('_')
            obj_class_id = int(self.graph_helper.object_dict.get_id(elements[1]))
            loc_class_id = int(self.graph_helper.object_dict.get_id(self.id2node[int(elements[2])]['class_name']))
            for _ in range(count):
                target_obj_class[pre_id] = obj_class_id
                target_loc_class[pre_id] = loc_class_id
                mask_goal_pred[pre_id] = 1.0
                pre_id += 1

        inputs.update({
            'affordance_matrix': self.graph_helper.obj1_affordance,
            'target_obj_class': target_obj_class,
            'target_loc_class': target_loc_class,
            'mask_goal_pred': mask_goal_pred,
            'gt_goal': obj_class_id
        })

        inputs_tensor = {}
        for input_name, inp in inputs.items():
            inp_tensor = torch.tensor(inp).unsqueeze(0)
            if inp_tensor.type() == 'torch.DoubleTensor':
                inp_tensor = inp_tensor.float()
            inputs_tensor[input_name] = inp_tensor


        value, action, action_probs, rnn_state, out_dict = self.actor_critic.act(
            inputs_tensor,
            rnn_hxs,
            masks,
            deterministic=self.deterministic,
            epsilon=self.epsilon,
            action_indices=action_indices)

        self.hidden_state = rnn_state
        info_model = {}
        info_model['probs'] = action_probs
        info_model['value'] = value
        info_model['actions'] = action
        info_model['state_inputs'] = copy.deepcopy(inputs_tensor)
        info_model['num_objects'] = inputs['mask_object'].sum(-1)
        info_model['num_objects_action'] = inputs['mask_action_node'].sum(-1)

        info_model['visible_ids'] = [node[1] for node in visible_objects]

        aux_out = self.actor_critic.auxiliary_pred(out_dict)
        info_model['pred_goal'] = aux_out['pred_goal']
        info_model['pred_close'] = aux_out['pred_close']
        info_model['gt_close'] = inputs_tensor['gt_close']
        info_model['gt_goal'] = inputs_tensor['gt_goal']
        info_model['mask_nodes'] = inputs_tensor['mask_object']

        #############
        # DEBUGGING
        # This is for debugging
        if self.args.use_gt_actions:

            id_glass = 459
            perc_correct_actions = 1.0
            if len([edge for edge in observation['edges'] if
                    edge['from_id'] == 1 and edge['to_id'] == id_glass and edge['relation_type'] == 'CLOSE']) > 0:
                # Grab
                action_id = self.graph_helper.action_dict.get_id('grab')
            else:
                # Walk to
                if not self.args.simulator_type == 'python':
                    action_id = self.graph_helper.action_dict.get_id('walktowards')
                else:
                    action_id = self.graph_helper.action_dict.get_id('walk')

            object_id = [it for it, node in enumerate(visible_objects) if node[1] == id_glass][0]

            if random.random() < perc_correct_actions:
                info_model['actions'] = [torch.tensor(action_id)[None, None], torch.tensor(object_id)[None, None]]


        action_str, action_tried = self.get_action_instr(info_model['actions'], visible_objects, observation)
        info_model['action_tried'] = action_tried
        # print('ACTIONS', info_model['actions'], action_str, action_probs[0],
        #       'IDS', inputs_tensor['node_ids'][0, :4])
        return action_str, info_model


    def get_action_instr(self, action, visible_objects, current_graph):
        python_env = self.args.simulator_type == 'python'
        action_name = self.graph_helper.action_dict.get_el(action[0].item())
        object_id = action[1].item()

        (o1, o1_id) = visible_objects[object_id]
        if o1 == 'no_obj':
            o1 = None
        action = utils_rl_agent.can_perform_action(action_name, o1, o1_id, self.agent_id, current_graph, teleport=(not python_env))
        action_try = '{} [{}] ({})'.format(action_name, o1, o1_id)
        #print('{: <40} --> {}'.format(action_try, action))
        return action, action_try