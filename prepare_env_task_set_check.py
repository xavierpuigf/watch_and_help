import sys

sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')
import pdb
import pickle
import json
import random
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent
from arguments import get_args
from algos.arena import Arena
from utils import utils_goals

OBJ_LIST = ['plate', 'waterglass', 'wineglass', 'cutleryfork', 'cupcake', 'juice', 'pancake', 'poundcake', 'wine',
            'pudding', 'apple', 'coffeepot', 'book']


def check_graph(graph):
    id2node = {node['id']: node for node in graph['nodes']}
    nodes_to_check = [node['id'] for node in graph['nodes'] if node['class_name'] in OBJ_LIST]
    connected_edges = {id: [] for id in nodes_to_check}
    for edge in graph['edges']:
        if edge['from_id'] in nodes_to_check and edge['relation_type'] != 'CLOSE' and id2node[edge['to_id']][
            'category'] != 'Rooms':
            connected_edges[edge['from_id']].append(edge)
    success = True
    for node_id, edges in connected_edges.items():
        if len(edges) < 1:
            print(id2node[node_id])
            print([id2node[edge['to_id']]['class_name'] for edge in graph['edges'] if edge['from_id'] == node_id])
            pdb.set_trace()
            success = False
    return success


if __name__ == '__main__':
    args = get_args()
    args.num_per_apartment = 50
    args.mode = 'full'

    home_dir = '.'  # '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/vh_multiagent_models'

    random.seed(1)

    failed_ids = []

    env_task_set = []
    # for task in ['setup_table']:#, 'put_fridge', 'put_dishwasher', 'prepare_food', 'read_book']:

    #args.dataset_path = home_dir + '/initial_environments/data/init_envs/7apartmet_50_check.pik'
    args.dataset_path = home_dir + '/initial_environments/data/init_envs/test_env_set_30_posteccv.pik'

    data = pickle.load(open(args.dataset_path, 'rb'))

    for task_id, problem_setup in enumerate(data):
        pdb.set_trace()
        env_id = problem_setup['apartment'] - 1
        task_name = problem_setup['task_name']
        init_graph = problem_setup['init_graph']

        if not check_graph(init_graph):
            failed_ids.append(task_id)

        goal = problem_setup['goal'][task_name]

        goals = utils_goals.convert_goal_spec(task_name, goal, init_graph,
                                              exclude=['cutleryknife'])
        print('env_id:', env_id)
        print('task_name:', task_name)
        print('goals:', goals)

        task_goal = {}
        for i in range(2):
            task_goal[i] = goals

        env_task_set.append({'task_id': task_id, 'task_name': task_name, 'env_id': env_id, 'init_graph': init_graph,
                             'task_goal': task_goal,
                             'level': 0,
                             'init_rooms': random.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)})

    pickle.dump(env_task_set, open('initial_environments/data/init_envs/env_task_set_30_check_posteccv.pik', 'wb'))
    print('failed_ids:', failed_ids)