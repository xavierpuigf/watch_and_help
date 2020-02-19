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
sys.argv = ['-f']

from agents import MCTS_agent, PG_agent
from envs.envs import UnityEnv


# Options, should go as argparse arguments
agent_type = 'MCTS' # PG/MCTS
simulator_type = 'unity' # unity/python
dataset_path = '../dataset_toy4/init_envs/'


def convert_goal_spec(task_name, goal, state):
    goals = {}
    containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] in ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']]
    id2node = {node['id']: node for node in state['nodes']}
    for key_count in goal:
        key = list(key_count.keys())[0]
        count = key_count[key]
        elements = key.split('_') 
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

def rollout_from_json(info):
        num_entries = len(info)
        count = 0

        while count < num_entries:

                info_entry = info[count]

                count += 1
                scene_index, _, graph_index = info_entry['env_path'].split('.')[0][len('TrimmedTestScene'):].split('_')
                path_init_env = '{}/{}'.format(dataset_path, info_entry['env_path'])
                state = load_graph_dict(path_init_env)['init_graph']
                goals = info_entry['goal']
                goal_index = info_entry['goal_index']


                env.reset(state, goals)

                env.to_pomdp()
                gt_state = env.vh_state.to_dict()

                agent_id = [x['id'] for x in gt_state['nodes'] if x['class_name'] == 'character'][0]

                print("{} / {}    ###(Goal: {} in scene{}_{})".format(count, num_entries, goals, scene_index, graph_index))

                if agent_type == 'PG':
                        agent = PG_agent(env,
                                                         max_episode_length=9,
                                                         num_simulation=1000.,
                                                         max_rollout_steps=5)

                        start = timeit.default_timer()
                        agent.rollout(state, goals)
                        end = timeit.default_timer()
                        print(end - start)

                elif agent_type == 'MCTS':
                        agent = MCTS_agent(env=env,
                                                             agent_id=agent_id,
                                                             max_episode_length=5,
                                                             num_simulation=100,
                                                             max_rollout_steps=5,
                                                             c_init=0.1,
                                                             c_base=1000000,
                                                             num_samples=1,
                                                             num_processes=1)
                else:
                        print('Agent {} not implemented'.format(agent_type))

                start = timeit.default_timer()
                history = agent.rollout(state, goals)

                end = timeit.default_timer()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--max-episode-length', type=int, default=100, help='Maximum episode length')
parser.add_argument('--agent-type', type=str, default='MCTS', help='Alice type: MCTS (default), PG')
parser.add_argument('--simulator-type', type=str, default='unity', help='Simulator type: unity (default), python')
parser.add_argument('--dataset-path', type=str, default='../initial_environments/data/init_envs/init7_20.p', help='Dataset path')

if __name__ == '__main__':
        args = parser.parse_args()
        print (' ' * 26 + 'Options')
        for k, v in vars(args).items():
                print(' ' * 26 + k + ': ' + str(v))
        
        # Non interactive rollout
        if args.simulator_type == 'python':
            env = gym.make('vh_graph-v0')
            print('Env created')


            info = [
                    {
                            'env_path': 'TrimmedTestScene1_graph_10.json',
                            'goal':  {0: ['findnode_2007']},
                            'goal_index': 0
                    }]

            rollout_from_json(info)
        else:
            num_agents = 1
            # data = pickle.load(open(args.dataset_path, 'rb'))
            # for problem_setup in data:
            #     env_id = problem_setup['apartment']
            #     task_name = problem_setup['task_name']
            #     init_graph = problem_setup['init_graph']
            #     goal = problem_setup['goal'][task_name]
            #     if task_name == 'setup_table':
            #         break
            unity_env = UnityEnv(env_id=4,
                                 num_agents=num_agents, 
                                 max_episode_length=100)
            
            # goals = convert_goal_spec(task_name, goal)
            # print('env_id:', env_id)
            # print('task_name:', task_name)
            # print('goals:', goals)


            ## ------------------------------------------------------------------------------
            ## Preparing the goal
            ## ------------------------------------------------------------------------------
            graph = unity_env.get_graph()
            # # glasses_id = [node['id'] for node in graph['nodes'] if 'wineglass' in node['class_name']]
            # # # print(glasses_id)
            # # # print([edge for edge in graph['edges'] if edge['from_id'] in glasses_id])
            table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
            print(table_id)
            # # goals = ['put_{}_{}'.format(glass_id, table_id) for glass_id in glasses_id][:2]
            # goals = {'on_{}_{}'.format('wineglass', table_id): 2}
            task_name = 'clean_table'
            goal = {task_name: [{'take_plate_off_{}'.format(table_id): 2}]}
            goals = convert_goal_spec(task_name, goal[task_name], graph)

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

            task_goal = {}
            for i in range(2):
                task_goal[i] = goals

            ## reset unity environment based on tthe goal
            unity_env.reset(graph, task_goal)


            if num_agents==1:
                unity_env.agents[unity_env.system_agent_id].run(graph, task_goal, single_agent=True)

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
                                                             num_processes=1)

                ## ------------------------------------------------------------------------------
                ## run your agent
                ## ------------------------------------------------------------------------------
                my_agent.run(graph, task_goal)
