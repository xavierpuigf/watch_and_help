import gym
import scipy.special
import ipdb
import sys
import json
import random
import numpy as np
import cProfile
import pdb
import timeit
import os
import argparse

# home_path = '/Users/xavierpuig/Desktop/MultiAgentBench/'
home_path = os.getcwd()
home_path = '/'.join(home_path.split('/')[:-2])

sys.path.append(home_path+'/vh_mdp')
sys.path.append(home_path+'/virtualhome')
sys.path.append(home_path+'/vh_multiagent_models')

import utils
from simulation.evolving_graph.utils import load_graph_dict
from profilehooks import profile
import pickle

from agents import MCTS_agent, PG_agent
from envs.envs import UnityEnv


# Options, should go as argparse arguments
agent_type = 'MCTS' # PG/MCTS
simulator_type = 'unity' # unity/python
dataset_path = '../dataset_toy4/init_envs/'


def convert_goal_spec(task_name, goal, state, exclude=[]):
    goals = {}
    containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] in ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']]
    id2node = {node['id']: node for node in state['nodes']}
    for key_count in goal:
        key = list(key_count.keys())[0]
        count = key_count[key]
        elements = key.split('_') 
        print(elements)
        if elements[1] in exclude: continue
        if task_name in ['setup_table', 'prepare_food']:
            predicate = 'on_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name in ['put_dishwasher', 'put_fridge']:
            predicate = 'inside_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name == 'clean_table':
            predicate = 'offOn'
            predicate = 'offOn_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
            # for edge in state['edges']:
            #     if edge['relation_type'] == 'ON' and edge['to_id'] == int(elements[3]) and id2node[edge['from_id']]['class_name'] == elements[1]:
            #         container = random.choice(containers)
            #         predicate = '{}_{}_{}'.format('on' if container[1] == 'kitchencounter' else 'inside', edge['from_id'], container[0])
            #         goals[predicate] = 1
        elif task_name == 'unload_dishwahser':
            predicate = 'offInside'
            predicate = 'offOn_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name == 'read_book':
            if elements[0] == 'holds':
                predicate = 'holds_{}_{}'.format('book', 1)
            elif elements[0] == 'sit':
                predicate = 'sit_{}_{}'.format(1, elements[1])
            else:
                predicate = 'on_{}_{}'.format(elements[1], elements[3])
                # count = 0
            goals[predicate] = count
        elif task_name == 'watch_tv':
            if elements[0] == 'holds':
                predicate = 'holds_{}_{}'.format('remotecontrol', 1)
            elif elements[0] == 'turnOn':
                predicate = 'turnOn_{}_{}'.format(elements[1], 1)
            elif elements[0] == 'sit':
                predicate = 'sit_{}_{}'.format(1, elements[1])
            else:
                predicate = 'on_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        else:
            predicate = key 
            goals[predicate] = count
        
    return goals


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--max-episode-length', type=int, default=250, help='Maximum episode length')
parser.add_argument('--agent-type', type=str, default='MCTS', help='Alice type: MCTS (default), PG')
parser.add_argument('--simulator-type', type=str, default='unity', help='Simulator type: python (default), unity')
# parser.add_argument('--dataset-path', type=str, default='../initial_environments/data/init_envs/init7_100_simple.p', help='Dataset path')
# parser.add_argument('--record-dir', type=str, default='../record/init7_100_same_room_simple', help='Record directory')
parser.add_argument('--recording', action='store_true', default=False, help='True - recording frames')
parser.add_argument('--num-per-apartment', type=int, default=10, help='Maximum #episodes/apartment')
parser.add_argument('--task', type=str, default='setup_table', help='Task name')
parser.add_argument('--mode', type=str, default='simple', help='Task name')
parser.add_argument('--port', type=int, default=8092, help='port')
parser.add_argument('--display', type=str, default='2', help='display')
parser.add_argument('--use-editor', action='store_true', default=False, help='Use unity editor')


if __name__ == '__main__':
        args = parser.parse_args()
        print (' ' * 26 + 'Options')
        for k, v in vars(args).items():
                print(' ' * 26 + k + ': ' + str(v))
        args.dataset_path = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/vh_multiagent_models/analysis/info_demo_scenes.json'
        args.record_dir = '../initial_environments/data/init_envs'
        
        data = json.load(open(args.dataset_path, 'r'))
        train_set = []
        for (task_id, task_key) in enumerate(data['split']['train']):
            env_set = data['goal_dict'][task_key]
            goal_class = env_set[0]
            json_file_list = env_set[1]

            for json_file in json_file_list:
                json_path = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/vh_multiagent_models' + \
                    json_file[2:]
                content = json.load(open(json_path, 'r'))
                init_unity_graph = content['init_unity_graph']
                init_graph = {}
                init_graph['nodes'] = [node for node in init_unity_graph['nodes'] if node['id'] not in [1, 2]]
                init_graph['edges'] = [edge for edge in init_unity_graph['edges'] if edge['from_id'] not in [1, 2] and edge['to_id'] not in [1, 2]]
                env_id = content['env_id']
                task_name = content['task_name']
                goals = content['goals']
                print('env_id:', env_id)
                print('task_name:', task_name)
                print('goals:', goals)
                task_goal = {}
                for i in range(2):
                    task_goal[i] = goals
                train_set.append({'task_id': task_id, 
                                  'task_name': task_name, 
                                  'env_id': env_id, 
                                  'init_graph': init_graph, 
                                  'task_goal': task_goal,
                                  'goal_class': goal_class,
                                  'level': 1, 
                                  'init_rooms': [0, 0],
                                  'json_file': json_file})
        pickle.dump(train_set, open(args.record_dir + '/train_demo_set.pik', 'wb'))

        test_set = []
        for (task_id, task_key) in enumerate(data['split']['test']):
            env_set = data['goal_dict'][task_key]
            goal_class = env_set[0]
            json_file_list = env_set[1]

            for json_file in json_file_list:
                json_path = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/vh_multiagent_models' + \
                    json_file[2:]
                content = json.load(open(json_path, 'r'))
                init_unity_graph = content['init_unity_graph']
                init_graph = {}
                init_graph['nodes'] = [node for node in init_unity_graph['nodes'] if node['id'] not in [1, 2]]
                init_graph['edges'] = [edge for edge in init_unity_graph['edges'] if edge['from_id'] not in [1, 2] and edge['to_id'] not in [1, 2]]
                env_id = content['env_id']
                task_name = content['task_name']
                goals = content['goals']
                print('env_id:', env_id)
                print('task_name:', task_name)
                print('goals:', goals)
                task_goal = {}
                for i in range(2):
                    task_goal[i] = goals
                test_set.append({'task_id': task_id, 
                                  'task_name': task_name, 
                                  'env_id': env_id, 
                                  'init_graph': init_graph, 
                                  'task_goal': task_goal,
                                  'goal_class': goal_class,
                                  'level': 1, 
                                  'init_rooms': [0, 0],
                                  'json_file': json_file})
        pickle.dump(test_set, open(args.record_dir + '/test_demo_set.pik', 'wb'))
