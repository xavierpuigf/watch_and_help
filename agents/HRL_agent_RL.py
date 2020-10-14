import torch
from models import actor_critic, actor_critic_hl_mcts
from gym import spaces
from utils import utils_rl_agent
import numpy as np
import pdb
import ipdb
import copy
from . import belief
from envs.graph_env import VhGraphEnv
import random

import numpy as np
import random
import time
import math
import copy
import importlib
import json
import multiprocessing
import ipdb
import pickle
from pathlib import Path

#
from MCTS import *

import sys

sys.path.append('..')
from utils import utils_environment as utils_env


def find_heuristic(agent_id, char_index, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    id2node = {node['id']: node for node in env_graph['nodes']}
    containerdict = {edge['from_id']: edge['to_id'] for edge in env_graph['edges'] if edge['relation_type'] == 'INSIDE'}
    target = int(object_target.split('_')[-1])
    observation_ids = [x['id'] for x in observations['nodes']]
    try:
        room_char = [edge['to_id'] for edge in env_graph['edges'] if
                     edge['from_id'] == agent_id and edge['relation_type'] == 'INSIDE'][0]
    except:
        print('Error')
        # ipdb.set_trace()

    action_list = []
    cost_list = []
    while target not in observation_ids:
        try:
            container = containerdict[target]
        except:
            print(id2node[target])
            # ipdb.set_trace()
        # If the object is a room, we have to walk to what is insde

        if id2node[container]['category'] == 'Rooms':
            action_list = [('walk', (id2node[target]['class_name'], target), None)] + action_list
            cost_list = [0.5] + cost_list

        elif 'CLOSED' in id2node[container]['states'] or ('OPEN' not in id2node[container]['states']):
            action = ('open', (id2node[container]['class_name'], container), None)
            action_list = [action] + action_list
            cost_list = [0.05] + cost_list

        target = container

    ids_character = [x['to_id'] for x in observations['edges'] if
                     x['from_id'] == agent_id and x['relation_type'] == 'CLOSE'] + \
                    [x['from_id'] for x in observations['edges'] if
                     x['to_id'] == agent_id and x['relation_type'] == 'CLOSE']

    if target not in ids_character:
        # If character is not next to the object, walk there
        action_list = [('walk', (id2node[target]['class_name'], target), None)] + action_list
        cost_list = [1] + cost_list

    return action_list, cost_list


def grab_heuristic(agent_id, char_index, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if (
                (edge['from_id'] == agent_id and edge['to_id'] == target_id) or (
                    edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if
                       (edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('grab', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, env_graph, simulator,
                                                  object_target)
        return find_actions + target_action, find_costs + cost


def turnOn_heuristic(agent_id, char_index, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if (
                (edge['from_id'] == agent_id and edge['to_id'] == target_id) or (
                    edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if
                       (edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('switchon', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, env_graph, simulator,
                                                  object_target)
        return find_actions + target_action, find_costs + cost


def sit_heuristic(agent_id, char_index, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if (
                (edge['from_id'] == agent_id and edge['to_id'] == target_id) or (
                    edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    on_ids = [edge['to_id'] for edge in env_graph['edges'] if
              (edge['from_id'] == agent_id and 'ON' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in on_ids:
        target_action = [('sit', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, env_graph, simulator,
                                                  object_target)
        return find_actions + target_action, find_costs + cost


def put_heuristic(agent_id, char_index, env_graph, simulator, target):
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]

    if sum([1 for edge in observations['edges'] if
            edge['from_id'] == target_grab and edge['to_id'] == target_put and edge['relation_type'] == 'ON']) > 0:
        # Object has been placed
        pdb.set_trace()
        return [], []

    if sum([1 for edge in observations['edges'] if
            edge['to_id'] == target_grab and edge['from_id'] != agent_id and 'HOLD' in edge['relation_type']]) > 0:
        # Object has been placed
        return None, None

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    id2node = {node['id']: node for node in env_graph['nodes']}
    target_grabbed = len([edge for edge in env_graph['edges'] if
                          edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'] and edge[
                              'to_id'] == target_grab]) > 0

    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, env_graph, simulator,
                                                   'grab_' + str(target_node['id']))
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == 'walk':
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]['category'] == 'Rooms':
                    object_diff_room = id_room

        env_graph_new = copy.deepcopy(env_graph)

        if object_diff_room:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if
                                      edge['to_id'] != agent_id and edge['from_id'] != agent_id]
            env_graph_new['edges'].append({'from_id': agent_id, 'to_id': object_diff_room, 'relation_type': 'INSIDE'})

        else:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if
                                      (edge['to_id'] != agent_id and edge['from_id'] != agent_id) or edge[
                                          'relation_type'] == 'INSIDE']
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, env_graph_new, simulator,
                                               'find_' + str(target_node2['id']))
    action = [('putback', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost = [0.05]
    res = grab_obj1 + find_obj2 + action
    cost_list = cost_grab_obj1 + cost_find_obj2 + cost

    # print(res, target)
    if len(res) == 0:
        pdb.set_trace()
    return res, cost_list

def open_heuristic(agent_id, char_index, env_graph, simulator, target):
    # observations = simulator.get_observations(env_graph, char_index=char_index)

    target_put = target.split('_')[-1]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, env_graph, simulator,
                                               'find_' + str(target_node2['id']))
    action_open = [('open', (target_node2['class_name'], target_put))]

    res = find_obj2 + action_open
    cost_list = []

    return res, cost_list

def putIn_heuristic(agent_id, char_index, env_graph, simulator, target):
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]

    if sum([1 for edge in observations['edges'] if
            edge['from_id'] == target_grab and edge['to_id'] == target_put and edge['relation_type'] == 'ON']) > 0:
        # Object has been placed
        return [], []

    if sum([1 for edge in observations['edges'] if
            edge['to_id'] == target_grab and edge['from_id'] != agent_id and 'HOLD' in edge['relation_type']]) > 0:
        # Object has been placed
        return None, None

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    id2node = {node['id']: node for node in env_graph['nodes']}
    target_grabbed = len([edge for edge in env_graph['edges'] if
                          edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'] and edge[
                              'to_id'] == target_grab]) > 0

    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, env_graph, simulator,
                                                   'grab_' + str(target_node['id']))
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == 'walk':
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]['category'] == 'Rooms':
                    object_diff_room = id_room

        env_graph_new = copy.deepcopy(env_graph)

        if object_diff_room:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if
                                      edge['to_id'] != agent_id and edge['from_id'] != agent_id]
            env_graph_new['edges'].append({'from_id': agent_id, 'to_id': object_diff_room, 'relation_type': 'INSIDE'})

        else:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if
                                      (edge['to_id'] != agent_id and edge['from_id'] != agent_id) or edge[
                                          'relation_type'] == 'INSIDE']
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, env_graph_new, simulator,
                                               'find_' + str(target_node2['id']))
    target_put_state = target_node2['states']
    action_open = [('open', (target_node2['class_name'], target_put))]
    action_put = [('putin', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost_open = [0.05]
    cost_put = [0.05]

    remained_to_put = 0


    if 'CLOSED' in target_put_state or 'OPEN' not in target_put_state:
        res = grab_obj1 + find_obj2 + action_open + action_put
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost_open + cost_put
    else:
        res = grab_obj1 + find_obj2 + action_put
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost_put

    # print(res, target)
    return res, cost_list


class HRL_agent_RL:
    """
    RL for a single agent
    """
    def __init__(self, args, agent_id, char_index, graph_helper, deterministic=False, action_space=['open', 'pickplace'], seed=123):
        self.args = args
        self.mode = 'train' if not args.evaluation else 'test'
        self.agent_type = 'RL_MCTS_RL'
        self.max_num_objects = args.max_num_objects
        self.actions = []
        self.seed = seed
        self.objects1, self.objects2 = [], []
        # all objects that can be grabbed
        grabbed_obj = graph_helper.object_dict_types["objects_grab"]
        # all objects that can be opene
        container_obj = graph_helper.object_dict_types["objects_inside"]
        surface_obj = graph_helper.object_dict_types["objects_surface"]
        for act in action_space:
            if act == 'open':
                self.actions.append('open')
                self.objects1.append("None")
                self.objects2 += container_obj
            if act == 'pickplace':
                self.actions.append('pickplace')
                self.objects2 += container_obj + surface_obj
                self.objects1 += grabbed_obj

        self.objects1 = list(set(self.objects1))
        self.objects2 = list(set(self.objects2))

        # Goal locations
        #self.objects1 = ["cupcake", "apple"]
        self.objects2 = ["coffeetable", "kitchentable", "dishwasher", "fridge"]

        self.obj2_dict = {}
        self.obj2_dict = {}

        self.num_actions = len(self.actions)
        self.num_object_classes = graph_helper.num_classes
        self.num_states = graph_helper.num_states

        self.char_index = char_index
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None

        self.last_action_low_level = None
        self.last_action = None
        self.action_count = 0

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

        self.action_space = spaces.Tuple((
            spaces.Discrete(len(self.objects1)),
            spaces.Discrete(len(self.objects2))
        ))
        self.action_space_lowlevel = spaces.Tuple((spaces.Discrete(graph_helper.num_actions),
                                          spaces.Discrete(self.num_object_classes)))
        self.actor_critic = actor_critic_hl_mcts.ActorCritic(self.action_space, base_name=args.base_net,
                                                             base_kwargs=base_kwargs, seed=seed)
        self.actor_critic_low_level = actor_critic.ActorCritic(self.action_space_lowlevel, base_name=args.base_net,
                                                               base_kwargs=base_kwargs, seed=seed)
        self.actor_critic_low_level_put = actor_critic.ActorCritic(self.action_space_lowlevel, base_name=args.base_net,
                                                                   base_kwargs=base_kwargs, seed=seed)

        self.actor_critic.base.main.main.bad_transformer = False

        self.id2node = None
        self.hidden_state = self.init_hidden_state()
        self.hidden_state_low_level = self.init_hidden_state()
        self.hidden_state_low_level_put = self.init_hidden_state()
        self.put_policy = False

        if torch.cuda.is_available():
            self.actor_critic.cuda()
            self.actor_critic_low_level.cuda()
            self.actor_critic_low_level_put.cuda()

        self.previous_belief_graph = None

    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph['edges']:
            key = (edge['from_id'], edge['to_id'])
            if key not in edge_dict:
                edge_dict[key] = [edge['relation_type']]
                new_edges.append(edge)
            else:
                if edge['relation_type'] not in edge_dict[key]:
                    edge_dict[key] += [edge['relation_type']]
                    new_edges.append(edge)

        graph['edges'] = new_edges
        return graph

    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def init_hidden_state(self):
        h_state = torch.zeros(1, self.hidden_size)
        c_state = torch.zeros(1, self.hidden_size)
        return (h_state, c_state)

    def reset(self, observed_graph, gt_graph, task_goal={}, seed=0):
        self.action_count = 0
        self.belief = Belief.Belief(gt_graph, agent_id=self.agent_id, seed=seed)
        self.belief.sample_from_belief()
        graph_belief = self.sample_belief(observed_graph)  # self.env.get_observations(char_index=self.char_index))
        self.sim_env.reset(graph_belief, task_goal)
        self.sim_env.to_pomdp()

        self.id2node = {node['id']: node for node in gt_graph['nodes']}
        self.hidden_state = self.init_hidden_state()
        self.hidden_state_low_level = self.init_hidden_state()
        self.hidden_state_low_level_put = self.init_hidden_state()
        self.put_policy = False

    def evaluate(self, rollout):
        pass

    def get_action(self, observation, goal_spec, action_space_ids=None, action_indices=None, full_graph=None):

        # ipdb.set_trace()
        print("GETTING ACTION")
        full_graph = None
        if full_graph is not None:
            observation_belief = self.sample_belief(full_graph)
        else:
            observation_belief = self.sample_belief(observation)
        self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})


        rnn_hxs = self.hidden_state
        rnn_hxs_low_level = self.hidden_state_low_level
        rnn_hxs_low_level_put = self.hidden_state_low_level_put

        masks = torch.ones(rnn_hxs[0].shape).type(rnn_hxs[0].type())

        if torch.cuda.is_available():
            rnn_hxs = (rnn_hxs[0].cuda(), rnn_hxs[1].cuda())
            rnn_hxs_low_level = (rnn_hxs_low_level[0].cuda(), rnn_hxs_low_level[1].cuda())
            rnn_hxs_low_level_put = (rnn_hxs_low_level_put[0].cuda(), rnn_hxs_low_level_put[1].cuda())
            masks = masks.cuda()
        inputs, info = self.graph_helper.build_graph(observation,
                                                     include_edges=self.args.base_net == 'GNN',
                                                     action_space_ids=action_space_ids,
                                                     character_id=self.agent_id)
        visible_objects = info[-1]
        action_space_ids = info[-2]

        target_obj_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        target_loc_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        mask_goal_pred = [0.0] * 6

        pre_id = 0
        obj_pred_names, loc_pred_names = [], []

        ##############################
        #### Inputs high level policy
        ##############################

        for predicate, info in goal_spec.items():
            count, required, reward = info
            if count == 0 or not required or 'sit' in predicate:
                continue

            # if not (predicate.startswith('on') or predicate.startswith('inside')):
            #     continue

            elements = predicate.split('_')
            obj_class_id = int(self.graph_helper.object_dict.get_id(elements[1]))
            loc_class_id = int(self.graph_helper.object_dict.get_id(self.id2node[int(elements[2])]['class_name']))

            obj_pred_names.append(elements[1])
            loc_pred_names.append(self.id2node[int(elements[2])]['class_name'])
            for _ in range(count):
                try:
                    target_obj_class[pre_id] = obj_class_id
                    target_loc_class[pre_id] = loc_class_id
                    mask_goal_pred[pre_id] = 1.0
                    pre_id += 1
                except:
                    pdb.set_trace()


        inputs.update({
            'affordance_matrix': self.graph_helper.obj1_affordance,
            'target_obj_class': target_obj_class,
            'target_loc_class': target_loc_class,
            'mask_goal_pred': mask_goal_pred,
            'gt_goal': obj_class_id
        })


        inputs_ll = {x:y for x,y in inputs.items()}

        inputs_tensor = {}
        for input_name, inp in inputs.items():
            inp_tensor = torch.tensor(inp).unsqueeze(0)
            if inp_tensor.type() == 'torch.DoubleTensor':
                inp_tensor = inp_tensor.float()
            inputs_tensor[input_name] = inp_tensor


        # ipdb.set_trace()

        if self.action_count == 0:
            self.last_action_low_level = None
            self.hidden_state_low_level = self.init_hidden_state()
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
            info_model['action_space_ids'] = action_space_ids
            # next_action = info_model['actions']
            next_action = info_model['actions']
            self.last_action = info_model['actions']
        else:
            next_action = self.last_action
            info_model = {}
            info_model['action_space_ids'] = action_space_ids
            info_model['visible_ids'] = [node[1] for node in visible_objects]
            info_model['mcts_action'] = True
            info_model['actions'] = next_action

        info_model['obs'] = observation['nodes']
        ###############
        # END HL-Policy
        ###############

        # pdb.set_trace()
        action_str, action_tried, plan, predicate = self.get_action_instr(next_action, visible_objects, observation_belief)
        pred_name = predicate
        #pred_name = list(self.goal_spec.keys())[0]
        # pred_name = 'on_pudding_73'
        pred_goal_spec = {pred_name: [1, True, 1]}
        # ipdb.set_trace()
        # pdb.set_trace()
        if predicate is not None:
            if not self.last_action_low_level is None:
                # If action is walking we dont save
                # pdb.set_trace()
                if 'walk' not in self.last_action_low_level:
                    self.last_action_low_level = None
                else:
                    # If action is walking but we are close, we dongt save
                    target_id = int(self.last_action_low_level.split('(')[1][:-1])
                    if len([edge for edge in observation['edges'] if edge['from_id'] == self.agent_id and edge['to_id'] == target_id and edge['relation_type'] in ['CLOSE', 'INSIDE']]):
                        self.last_action_low_level = None

            if self.last_action_low_level is None:

                ##############################
                #### Inputs low level policy
                ##############################
                pre_id = 0
                target_obj_class_pred = [self.graph_helper.object_dict.get_id('no_obj')] * 6
                target_loc_class_pred = [self.graph_helper.object_dict.get_id('no_obj')] * 6
                mask_goal_pred_pred = [0.0] * 6

                for predicate, info in pred_goal_spec.items():
                    count, required, reward = info
                    if count == 0 or not required:
                        continue

                    # if not (predicate.startswith('on') or predicate.startswith('inside')):
                    #     continue

                    elements = predicate.split('_')
                    # print('PRED_OB', elements[1])
                    obj_class_id = int(self.graph_helper.object_dict.get_id(elements[1]))
                    if elements[2].isdigit():
                        loc_class_id = int(self.graph_helper.object_dict.get_id(self.id2node[int(elements[2])]['class_name']))
                    else:
                        loc_class_id = int(self.graph_helper.object_dict.get_id(elements[2]))
                    for _ in range(count):
                        target_obj_class_pred[pre_id] = obj_class_id
                        target_loc_class_pred[pre_id] = loc_class_id
                        mask_goal_pred_pred[pre_id] = 1.0
                        pre_id += 1

                # ipdb.set_trace()
                # ipdb.set_trace()

                inputs_ll.update({
                    'affordance_matrix': self.graph_helper.obj1_affordance,
                    'target_obj_class': target_obj_class_pred,
                    'target_loc_class': target_loc_class_pred,
                    'mask_goal_pred': mask_goal_pred_pred,
                    'gt_goal': obj_class_id
                })
                # ipdb.set_trace()
                #### END INPUTS LOW LEVEL ####
                # ipdb.set_trace()
                inputs_tensor_ll = {}
                for input_name, inp in inputs_ll.items():
                    inp_tensor = torch.tensor(inp).unsqueeze(0)
                    if inp_tensor.type() == 'torch.DoubleTensor':
                        inp_tensor = inp_tensor.float()
                    inputs_tensor_ll[input_name] = inp_tensor

                if self.put_policy is False:
                    value_ll, action_ll, action_probs_ll, rnn_state_ll, out_dict_ll = self.actor_critic_low_level.act(inputs_tensor_ll, rnn_hxs_low_level, masks)
                    self.hidden_state_low_level_put = self.init_hidden_state()
                    self.hidden_state_low_level = rnn_state_ll
                else:
                    value_ll, action_ll, action_probs_ll, rnn_state_ll, out_dict_ll = self.actor_critic_low_level_put.act(
                        inputs_tensor_ll, rnn_hxs_low_level_put, masks)
                    self.hidden_state_low_level_put = rnn_state_ll
                    self.hidden_state_low_level = self.init_hidden_state()

                action_str, action_tried = self.get_action_instr_low_level(action_ll, visible_objects, observation)
                if action_str is not None:
                    print(action_str)
                    if self.put_policy and '[put' in action_str:
                        self.put_policy = False
                    elif 'grab' in action_str:
                        self.put_policy = True
                    # ipdb.set_trace()
                    # print("LOW LEVEL ACTION" , action_str)
                    if action_str is not None:
                        self.last_action_low_level = action_str

            else:
                # print("SAME")

                action_str = self.last_action_low_level


            self.action_count += 1
            if len(plan) == 1 or self.action_count >= self.args.num_steps_mcts:
                self.action_count = 0

        else:
            # print("Plan: ", plan, action_tried)
            self.action_count = 0


        # print("Action low level", action_str, predicate)
        info_model['action_tried'] = action_tried
        info_model['predicate'] = predicate
        # print('ACTIONS', info_model['actions'], action_str, action_probs[0],
        #       'IDS', inputs_tensor['node_ids'][0, :4])


        return action_str, info_model


    def get_action_instr_low_level(self, action, visible_objects, current_graph):
        python_env = self.args.simulator_type == 'python'
        action_name = self.graph_helper.action_dict.get_el(action[0].item())
        object_id = action[1].item()

        (o1, o1_id) = visible_objects[object_id]
        if o1 == 'no_obj':
            o1 = None
        action = utils_rl_agent.can_perform_action(action_name, o1, o1_id, self.agent_id, current_graph, teleport=self.args.teleport)
        action_try = '{} [{}] ({})'.format(action_name, o1, o1_id)
        #print('{: <40} --> {}'.format(action_try, action))
        return action, action_try

    def get_action_instr(self, action, visible_objects, current_graph):
        # Build action"
        if self.objects1[action[0].item()] == "None":
            # Open action, open a new object that was not open before
            if self.objects2[action[1].item()] not in self.graph_helper.object_dict_types["objects_inside"]:
                return None, "open_{}".format(self.objects2[action[1].item()]), [], "open_{}".format(self.objects2[action[1].item()])

            target_id = [node['id'] for node in current_graph['nodes'] if node['class_name'] == self.objects2[action[1].item()] and node['states'] == 'CLOSED']
            if len(target_id) == 0:
                return None,  "open_{}".format(self.objects2[action[1].item()]), [], "open_{}".format(self.objects2[action[1].item()])
            target_goal = 'open_{}'.format(target_id[0])

            actions, _ = open_heuristic(self.agent_id, 0, current_graph, self.sim_env, target_goal)
        else:
            # Pick ans place
            obj_name = self.objects1[action[0].item()]
            container_name = self.objects2[action[1].item()]
            container_id = [node['id'] for node in current_graph['nodes'] if node['class_name'] == self.objects2[action[1].item()]]
            if len(container_id) == 0:
                return None, 'put_{}_{}'.format(obj_name, container_name), [], 'put_{}_{}'.format(obj_name, container_name)
            obj_rel_container = [edge['from_id'] for edge in current_graph['edges'] if edge['to_id'] == container_id[0]
                                 and edge['relation_type'] in ['ON', 'INSIDE']]

            # Objects that are not there
            object_id = [node['id'] for node in current_graph['nodes'] if node['class_name'] == self.objects1[action[0].item()] and
                         node['id'] not in obj_rel_container]
            if len(object_id) == 0:

                return None, 'put_{}_{}'.format(obj_name, container_name), [], 'put_{}_{}'.format(obj_name, container_name)


            # Select the shortest task
            min_cost = 0
            actions = None
            for obj_id in range(len(object_id)):
                target_goal = "put_{}_{}".format(object_id[obj_id], container_id[0])
                # print("Heurisitc: ", target_goal)
                if container_name in self.graph_helper.object_dict_types['objects_surface']:
                    actions_curr, cost = put_heuristic(self.agent_id, self.char_index, current_graph, self.sim_env, target_goal)
                else:
                    actions_curr, cost = putIn_heuristic(self.agent_id, self.char_index, current_graph, self.sim_env, target_goal)

                if cost is None or len(cost) == 0:
                    continue
                curr_cost_plan = sum(cost)
                if obj_id == 0 or curr_cost_plan < min_cost:
                    min_cost = curr_cost_plan
                    actions = actions_curr




        if actions is None:
            return None, 'put_{}_{}'.format(obj_name, container_name), actions, 'put_{}_{}'.format(obj_name, container_name)
        action_name = actions[0][0]
        if 'put' in action_name:
            obj_id_action = 2
        else:
            obj_id_action = 1
        o1, o1_id = actions[0][obj_id_action]
        action_name = action_name.replace("walk", "walktowards")

        action = utils_rl_agent.can_perform_action(action_name, o1, o1_id, self.agent_id, current_graph,
                                                   graph_helper=self.graph_helper,
                                                   teleport=self.args.teleport)
        action_try = '{} [{}] ({})'.format(action_name, o1, o1_id)
        #print('{: <40} --> {}'.format(action_try, action))
        # print(action_try, action)
        return action, action_try, actions, 'put_{}_{}'.format(obj_name, container_name)
