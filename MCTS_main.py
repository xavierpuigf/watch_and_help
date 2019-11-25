import gym
import sys
# sys.path.append('../vh_mdp')
# sys.path.append('../virtualhome')
import vh_graph
from vh_graph.envs import belief
import utils_viz
import utils
import json
import numpy as np
from simulation.evolving_graph.utils import load_graph_dict
sys.argv = ['-f']

from agents import MCTS_agent

import timeit

print('Imports done')
env = gym.make('vh_graph-v0')
print('Env created')

path_init_env = 'dataset_toy4/init_envs/TrimmedTestScene6_graph_35.json'
# goal_name = {0:'(HOLDS_RH character[240] plate[2005])'}
goal_name = {0: 'findnode_2038'}
state = load_graph_dict(path_init_env)['init_graph']
env.reset(state, goal_name)
env.to_pomdp()
gt_state = env.vh_state.to_dict()

id_agent = [x['id'] for x in gt_state['nodes'] if x['class_name'] == 'character'][0]
id_goal = 2038
goal_str = 'findnode_{}'.format(id_goal)
print([n for n in gt_state['nodes'] if n['category'] == 'Rooms'])
print([n for n in gt_state['nodes'] if n['id'] == id_goal])
print([[(n['id'], n['class_name']) for n in gt_state['nodes'] if n['id'] == e['from_id']] for e in gt_state['edges'] if 246 in e.values()])
print([e for e in gt_state['edges'] if id_goal in e.values()])
print([e for e in gt_state['edges'] if id_agent in e.values()])

np.random.seed(1)
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


agent = MCTS_agent(env=env,
                   # sim_env=sim_env, 
                   # bel=bel,
                   max_episode_length=100,
                   num_simulation=1000, 
                   max_rollout_steps=5, 
                   c_init=1.25, 
                   c_base=1000000,
                   num_samples=1,
                   num_processes=1)
start = timeit.default_timer()
agent.rollout(state, goal_name)
end = timeit.default_timer()
print(end - start)

# agent = MCTS_agent(env=env, 
#                    max_episode_length=5,
#                    num_simulation=500, 
#                    max_rollout_steps=5, 
#                    c_init=0.1, 
#                    c_base=1000000,
#                    num_samples=1,
#                    num_processes=1)
# start = timeit.default_timer()
# agent.rollout(state, goal_name)
# end = timeit.default_timer()
# print(end - start)