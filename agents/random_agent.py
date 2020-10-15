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
import os


from . import belief
from envs.graph_env import VhGraphEnv
from utils import utils_rl_agent

import sys
sys.path.append('..')
from utils import utils_environment as utils_env


def clean_graph(state, goal_spec, last_opened):
    new_graph = {}
    # get all ids
    ids_interaction = []
    nodes_missing = []
    for predicate in goal_spec:
        elements = predicate.split('_')
        nodes_missing += [int(x) for x in elements if x.isdigit()]
        for x in elements[1:]:
            if x.isdigit():
                nodes_missing += [int(x)]
            else:
                nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == x]
    nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == 'character' or node['category'] in ['Rooms', 'Doors']]

    id2node = {node['id']: node for node in state['nodes']}
    # print([node for node in state['nodes'] if node['class_name'] == 'kitchentable'])
    # print(id2node[235])
    # ipdb.set_trace()
    inside = {}
    for edge in state['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] not in inside.keys():
                inside[edge['from_id']] = []
            inside[edge['from_id']].append(edge['to_id'])
    
    while (len(nodes_missing) > 0):
        new_nodes_missing = []
        for node_missing in nodes_missing:
            if node_missing in inside:
                new_nodes_missing += [node_in for node_in in inside[node_missing] if node_in not in ids_interaction]
            ids_interaction.append(node_missing)
        nodes_missing = list(set(new_nodes_missing))

    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    # for clean up tasks, add places to put objects to
    augmented_class_names = []
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['dishwasher', 'kitchentable']:
                augmented_class_names += ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']
                break
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['sofa', 'chair']:
                augmented_class_names += ['coffeetable']
                break
    containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] in augmented_class_names]
    for obj_id in containers:
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)


    new_graph = {
            "edges": [edge for edge in state['edges'] if edge['from_id'] in ids_interaction and edge['to_id'] in ids_interaction],
            "nodes": [id2node[id_node] for id_node in ids_interaction]
    }

    return new_graph


class Random_agent:
    """
    Random agent
    """
    def __init__(self, agent_id, char_index,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, recursive=False,
                 num_samples=1, num_processes=1, comm=None, logging=False, logging_graphs=False, seed=None):
        self.agent_type = 'Random'
        self.verbose = False
        self.recursive = recursive

        #self.env = unity_env.env
        if seed is None:
            seed = random.randint(0,100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.agent_id = agent_id
        self.char_index = char_index
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes
        
        self.previous_belief_graph = None
        self.verbose = False

        # Indicates whether there is a unity simulation
        self.comm = comm


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
        # for edge in new_graph['edges']:
        #     if edge['from_id'] == 272 and edge['to_id'] == 271:
        #         ipdb.set_trace()
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

        # # TODO: probably these 2 cases are not needed
        # if self.previous_belief_graph is None:
        #     self.belief.reset_belief()
        #     new_graph = self.belief.sample_from_belief()
        #     new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        #     self.previous_belief_graph = new_graph
        # else:
        #     new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        #     self.previous_belief_graph = new_graph

        # return new_graph

    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
        edges = [edge for edge in graph['edges'] if edge['from_id'] == char_id]
        print('Character:')
        print(edges)
        print('---')


    def get_action(self, obs, goal_spec, opponent_subgoal=None):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/../dataset/object_info_small.json', 'r') as f:
            content = json.load(f)
        object_names = []
        for obj in content.values():
            object_names += obj
        object_names += ['bathroom', 'bedroom', 'kitchen', 'livingroom', 'character']

        self.sample_belief(obs)
        self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})

        action_name = random.choice(['walktowards', 'grab', 'put', 'open'])
        if action_name == 'walk':
            objects = [(node['class_name'], node['id']) for node in obs['nodes'] if node['class_name'] in object_names]
        elif action_name == 'grab':
            objects = [(node['class_name'], node['id']) for node in obs['nodes'] if node['class_name'] in content['objects_grab']]
        elif action_name == 'put':
            objects = [(node['class_name'], node['id']) for node in obs['nodes'] if node['class_name'] in content['objects_surface'] + content['objects_inside']]
        else:
            objects = [(node['class_name'], node['id']) for node in obs['nodes'] if node['class_name'] in content['objects_inside']]

        if len(objects) == 0:
            action = None
        else:
            selected_object = random.choice(objects)
            o1, o1_id = selected_object[0], selected_object[1]

            action = utils_rl_agent.can_perform_action(action_name, o1, o1_id, self.agent_id, obs, teleport=False)


        if self.logging:
            info = {
                'plan': [action],
                'belief': copy.deepcopy(self.belief.edge_belief),
                'belief_graph': copy.deepcopy(self.sim_env.vh_state.to_dict())
            }
            if self.logging_graphs:
                info.update(
                    {'obs': obs['nodes']})
        else:
            info = {}

        # return None, info
        return action, info

    def reset(self, observed_graph, gt_graph, task_goal, seed=0, simulator_type='python', is_alice=False):

        self.last_action = None
        self.last_subgoal = None
        """TODO: do no need this?"""

        self.previous_belief_graph = None
        self.belief = belief.Belief(gt_graph, agent_id=self.agent_id, seed=seed)
        # print("set")
        self.belief.sample_from_belief()
        graph_belief = self.sample_belief(observed_graph)
        self.sim_env.reset(graph_belief, task_goal)
        self.sim_env.to_pomdp()
        

