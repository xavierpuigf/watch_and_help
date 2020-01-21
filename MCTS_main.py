import gym
import sys
sys.path.append('../vh_mdp')
sys.path.append('../virtualhome')
import vh_graph
from vh_graph.envs import belief
import utils_viz
import utils
import json
import random
import numpy as np
from simulation.evolving_graph.utils import load_graph_dict
from simulation.unity_simulator import comm_unity as comm_unity

import pickle
sys.argv = ['-f']

from agents import MCTS_agent, PG_agent

import timeit

# Options, should go as argparse arguments
agent_type = 'MCTS' # PG/MCTS
simulator_type = 'python' # unity/python
dataset_path = '../dataset_toy4/init_envs/'


def rollout_from_json(info):
    num_entries = len(info)
    count = 0

    while count < num_entries:

        info_entry = info[count]

        count += 1
        scene_index, _, graph_index = info_entry['env_path'].split('.')[0][len('TrimmedTestScene'):].split('_')
        path_init_env = '{}/{}'.format(dataset_path, info_entry['env_path'])
        goals = info_entry['goal']
        goal_index = info_entry['goal_index']
        state = load_graph_dict(path_init_env)['init_graph']

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

def interactive_rollout():
    comm = comm_unity.UnityCommunication()
    comm.reset(0)
    agent = MCTS_agent(env=env,
                       agent_id=0,
                       max_episode_length=5,
                       num_simulation=100,
                       max_rollout_steps=5,
                       c_init=0.1,
                       c_base=1000000,
                       num_samples=1,
                       num_processes=1)

    task_goal = 'findnode_2007'
    while True:

        agent.get_action(graph, task_goal)
        input('Select next action')

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
        interactive_rollout()


