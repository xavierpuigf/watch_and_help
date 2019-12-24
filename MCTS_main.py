import gym
import sys
# sys.path.append('../vh_mdp')
# sys.path.append('../virtualhome')
import vh_graph
from vh_graph.envs import belief
import utils_viz
import utils
import json
import random
import numpy as np
from simulation.evolving_graph.utils import load_graph_dict
sys.argv = ['-f']

from agents import MCTS_agent, PG_agent

import timeit

print('Imports done')
env = gym.make('vh_graph-v0')
print('Env created')

with open('scenes_populated/info.json', 'r') as f:
    info = json.load(f)

num_entries = len(info)
count = 0

while count < num_entries:
    
    with open('log_progress.txt', 'r') as f:
        count = int(f.read())

    info_entry = info[count]

    count += 1
    #path_init_env = 'dataset_toy4/init_envs/TrimmedTestScene1_graph_10.json'
    scene_index, graph_index = info_entry['env_path'].split('.')[0][5:].split('_')

    path_init_env = 'scenes_populated/init_envs/{}'.format(info_entry['env_path'])
    goal_str = info_entry['goal']
    goal_name = {0: goal_str}
    goal_index = info_entry['goal_index']
    state = load_graph_dict(path_init_env)['init_graph']
    env.reset(state, goal_name)
    env.to_pomdp()
    gt_state = env.vh_state.to_dict()

    id_agent = [x['id'] for x in gt_state['nodes'] if x['class_name'] == 'character'][0]

    print("{} / {}    ###(Goal: {} in scene{}_{})".format(count, num_entries, goal_str, scene_index, graph_index))
    
    #print([n for n in gt_state['nodes'] if n['category'] == 'Rooms'])
    #print([n for n in gt_state['nodes'] if n['id'] == id_goal])
    #print([[(n['id'], n['class_name']) for n in gt_state['nodes'] if n['id'] == e['from_id']] for e in gt_state['edges'] if 246 in e.values()])
    #print([e for e in gt_state['edges'] if id_goal in e.values()])
    #print([e for e in gt_state['edges'] if id_agent in e.values()])

    # np.random.seed(1)
    # random.seed(1)

    # bel = belief.Belief(gt_state)
    # new_graph = bel.sample_from_belief()
    # sim_env = gym.make('vh_graph-v0')
    # sim_env.reset(new_graph, goal_name)
    # # sim_env.to_pomdp()
    # sim_state = sim_env.vh_state.to_dict()
    # # while True:
    # #     bel = belief.Belief(gt_state)
    # #     new_graph = bel.sample_from_belief()
    # #     sim_env = gym.make('vh_graph-v0')
    # #     sim_env.reset(new_graph, goal_name)
    # #     sim_env.to_pomdp()
    # #     sim_state = sim_env.vh_state.to_dict()
    # #     if 117 in [e['to_id'] for e in sim_state['edges'] if e['from_id'] == id_goal and e['relation_type'] == 'INSIDE']:
    # #         if 41 in [e['to_id'] for e in sim_state['edges'] if 117 in e.values() and e['relation_type'] == 'INSIDE']: 
    # #             break
    # # print(gt_state)
    # print('sim')
    # print([n for n in sim_state['nodes'] if n['category'] == 'Rooms'])
    # print([n for n in sim_state['nodes'] if n['id'] == id_goal])
    # print([[(n['id'], n['class_name']) for n in sim_state['nodes'] if n['id'] == e['from_id']] for e in sim_state['edges'] if 41 in e.values()])
    # print([e for e in sim_state['edges'] if id_goal in e.values()])
    # print([e for e in sim_state['edges'] if 284 in e.values() and e['relation_type'] == 'INSIDE'])
    # print([e for e in sim_state['edges'] if id_agent in e.values()])

    # env.step({0: '[walk] <dining_room> (41)'})
    # obs_graph = env.get_observations(0)
    # bel.update_from_gt_graph(obs_graph)
    # new_graph = bel.sample_from_belief()
    # sim_env.reset(new_graph, goal_name)


    # agent = MCTS_agent(env=env,
    #                    # sim_env=sim_env,
    #                    # bel=bel,
    #                    max_episode_length=100,
    #                    num_simulation=1000,
    #                    max_rollout_steps=5,
    #                    c_init=1.25,
    #                    c_base=1000000,
    #                    num_samples=1,
    #                    num_processes=1)



    """ agent = PG_agent(env,
                    max_episode_length=9,
                    num_simulation=1000.,
                    max_rollout_steps=5)

    start = timeit.default_timer()
    agent.rollout(state, goal_name)
    end = timeit.default_timer()
    print(end - start) """

    agent = MCTS_agent(env=env, 
                    max_episode_length=5,
                    num_simulation=100, 
                    max_rollout_steps=5, 
                    c_init=0.1, 
                    c_base=1000000,
                    num_samples=1,
                    num_processes=1)
    start = timeit.default_timer()
    agent.rollout(state, goal_name, scene_index, graph_index, goal_index)
    end = timeit.default_timer()
    #print(end - start) 

    with open('log_progress.txt', 'w') as f:
        f.write(str(count))
