import gym
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
        if elements[1] in exclude: continue
        if task_name == 'setup_table':
            predicate = 'on_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name in ['put_dishwasher', 'put_fridge']:
            predicate = 'in_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name == 'clean_table':
            for edge in state['edges']:
                if edge['relation_type'] == 'ON' and edge['to_id'] == int(elements[3]) and id2node[edge['from_id']]['class_name'] == elements[1]:
                    container = random.choice(containers)
                    predicate = '{}_{}_{}'.format('on' if container[1] == 'kitchencounter' else 'inside', edge['from_id'], container[0])
                    goals[predicate] = 1
        else:
            predicate = key 
            goals[predicate] = count
        
    return goals


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--max-episode-length', type=int, default=200, help='Maximum episode length')
parser.add_argument('--agent-type', type=str, default='MCTS', help='Alice type: MCTS (default), PG')
parser.add_argument('--simulator-type', type=str, default='unity', help='Simulator type: python (default), unity')
parser.add_argument('--dataset-path', type=str, default='../initial_environments/data/init_envs/init1_10.p', help='Dataset path')
parser.add_argument('--record-dir', type=str, default='../record', help='Record directory')


if __name__ == '__main__':
        args = parser.parse_args()
        print (' ' * 26 + 'Options')
        for k, v in vars(args).items():
                print(' ' * 26 + k + ': ' + str(v))
        
        num_agents = 1
        data = pickle.load(open(args.dataset_path, 'rb'))
        env_task_set = []
        for task_id, problem_setup in enumerate(data):
            env_id = problem_setup['apartment'] - 1
            task_name = problem_setup['task_name']
            init_graph = problem_setup['init_graph']
            goal = problem_setup['goal'][task_name]
            if task_name != 'setup_table':
                continue
            goals = convert_goal_spec(task_name, goal, init_graph, 
                                  exclude=[])
            # print('env_id:', env_id)
            # print('task_name:', task_name)
            # print('goals:', goals)

            task_goal = {}
            for i in range(2):
                task_goal[i] = goals

            env_task_set.append({'task_id': task_id, 'task_name': task_name, 'env_id': env_id, 'init_graph': init_graph, 'task_goal': task_goal,
                                'level': 0, 'init_rooms': [0, 0]})

        unity_env = UnityEnv(num_agents=num_agents, 
                             max_episode_length=args.max_episode_length,
                             simulator_type=args.simulator_type,
                             env_task_set=env_task_set,
                             logging=True,
                             recording=True,
                             record_dir=args.record_dir)
        
        steps_list, failed_tasks = [], []
        for episode_id in range(len(env_task_set)):
            unity_env.reset_MCTS(task_id=episode_id)

            ## ------------------------------------------------------------------------------
            ## Preparing the goal
            ## ------------------------------------------------------------------------------
            graph = unity_env.get_graph()
            # # glasses_id = [node['id'] for node in graph['nodes'] if 'wineglass' in node['class_name']]
            # # print(glasses_id)
            # # # # print([edge for edge in graph['edges'] if edge['from_id'] in glasses_id])
            # table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
            # print(table_id)
            # # # goals = ['put_{}_{}'.format(glass_id, table_id) for glass_id in glasses_id][:2]
            # # goals = {'on_{}_{}'.format('wineglass', table_id): 2}
            # # task_name = 'clean_table'
            # task_name = 'setup_table'
            # # goal = {task_name: [{'take_plate_off_{}'.format(table_id): 2}]}
            # goal = {task_name: [{'put_wineglass_on_{}'.format(table_id): 2}]}
            # goals = convert_goal_spec(task_name, goal[task_name], graph)

            # # # put dishes into the dish washer
            # # fridge_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'dishwasher'][0]
            # # obj_class_names = ['plate', 'dishbowl']
            # # # goals = {}
            # # for obj_class_name in obj_class_names:
            # #     count = len([node['id'] for node in graph['nodes'] if obj_class_name in node['class_name']])
            # #     if count == 0:
            # #         continue
            # #     goals['inside_{}_{}'.format(obj_class_name, fridge_id)] = count
            # # print('goals:', goals)

            # # put food into the fridge
            # fridge_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fridge'][0]
            # obj_class_names = ['cupcake']
            # # goals = {}
            # for obj_class_name in obj_class_names:
            #     count = len([node['id'] for node in graph['nodes'] if obj_class_name in node['class_name']])
            #     if count == 0:
            #         continue
            #     goals['inside_{}_{}'.format(obj_class_name, fridge_id)] = count
            # print('goals:', goals)

            

            ## reset unity environment based on the goal
            # unity_env.reset_alice(graph, task_goal)
            # unity_env.setup(graph, task_goal)
            # unity_env.reset()


            if num_agents==1:
                steps, finished = unity_env.agents[unity_env.system_agent_id].run(single_agent=True)
                if not finished:
                    failed_tasks.append(episode_id)
                else:
                    steps_list.append(steps)

            else:
                ## ------------------------------------------------------------------------------
                ## your agent, add your code here
                ## ------------------------------------------------------------------------------
                my_agent_id = unity_env.get_my_agent_id()
                my_agent = MCTS_agent(unity_env=unity_env,
                                     agent_id=my_agent_id,
                                     char_index=1,
                                     max_episode_length=5,
                                     num_simulation=100,
                                     max_rollout_steps=3,
                                     c_init=0.1,
                                     c_base=1000000,
                                     num_samples=1,
                                     num_processes=1,
                                     logging=True)

                ## ------------------------------------------------------------------------------
                ## run your agent
                ## ------------------------------------------------------------------------------
                my_agent.run()
        print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
        print('failed_tasks:', failed_tasks)
