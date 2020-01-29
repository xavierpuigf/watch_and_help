import gym
import ipdb
import sys
import json
import random
import numpy as np
import cProfile
import pdb
import timeit

home_path = '/Users/shuangli/Desktop/0mit/0research/0icml2020/1virtualhome/'
sys.path.append(home_path+'/vh_mdp')
sys.path.append(home_path+'/virtualhome_pkg')
sys.path.append(home_path+'/unified-agent')

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



if __name__ == '__main__':

    
    # Non interactive rollout
    if simulator_type == 'python':
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
        unity_env = UnityEnv(num_agents)
        
        ## ------------------------------------------------------------------------------
        ## Preparing the goal
        ## ------------------------------------------------------------------------------
        graph = unity_env.unity_simulator.get_graph()
        glasses_id = [node['id'] for node in graph['nodes'] if 'wineglass' in node['class_name']]
        table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
        goals = ['put_{}_{}'.format(glass_id, table_id) for glass_id in glasses_id][:2]
        task_goal = {}
        for i in range(2):
            task_goal[i] = goals


        if num_agents==1:
          unity_env.agents[unity_env.system_agent_id].run(graph, task_goal, single_agent=True)

        else:
          ## ------------------------------------------------------------------------------
          ## your agent, add your code here
          ## ------------------------------------------------------------------------------
          my_agent_id = unity_env.get_my_agent_id()
          my_agent = MCTS_agent(unity_env=unity_env,
                                 agent_id=my_agent_id,
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

