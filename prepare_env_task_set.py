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


if __name__ == '__main__':
    args = get_args()
    
    home_dir = '.'#'/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/vh_multiagent_models'

    random.seed(1)
    args.num_per_apartment = 40
    args.mode =
    env_task_set = []
    for task in ['setup_table', 'put_fridge', 'put_dishwasher', 'prepare_food', 'read_book']:
        args.dataset_path = home_dir + '/initial_environments/data/init_envs/init7_{}_{}_{}.pik'.format(task,
                                                                                            args.num_per_apartment,
                                                                                            args.mode)
        data = pickle.load(open(args.dataset_path, 'rb'))

        for task_id, problem_setup in enumerate(data):
            env_id = problem_setup['apartment'] - 1
            task_name = problem_setup['task_name']
            init_graph = problem_setup['init_graph']
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
                                 'level': 0, 'init_rooms': random.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)})

        pickle.dump(env_task_set, open('initial_environments/data/init_envs/env_task_set_{}_{}.pik'.format(args.num_per_apartment, args.mode), 'wb'))

