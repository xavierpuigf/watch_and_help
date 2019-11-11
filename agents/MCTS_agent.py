from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from pathlib import Path
import random
import time
import math
import copy
import pickle
import importlib
import multiprocessing

from MCTS import *


def sampler(sample_id, root_action, root_node, env, mcts, nb_steps, res):
    init_vh_state = env.vh_state
    init_state = env.state
    # print('init state:', init_state)
    # print('init action space:', env.get_action_space(init_state))
    # input('press any key ton continue...')
    if env.is_terminal(0, init_state):
        terminal = True
        res[sample_id] = None
        return
    if root_action is None:
        root_node = Node(id={root_action: [init_vh_state, init_state]},
                        num_visited=0,
                        sum_value=0,
                        is_expanded=False)
        curr_node = root_node
        next_root, action = mcts.run(curr_node, nb_steps)
    else:
        action, _, next_root = mcts.select_next_root(root_node)
    print(root_node.sum_value)
    # print(sample_id, res[sample_id])
    if sample_id is not None:
        res[sample_id] = action
    else:
        return action, next_root


class MCTS_agent:
    """
    MCTS for a single agent
    """
    def __init__(self, env, sim_env, bel, max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, num_samples=1, num_processes=1):
        self.env = env
        self.sim_env = sim_env
        self.bel = bel
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes
        self.mcts = MCTS(env, max_episode_length, num_simulation, max_rollout_steps, c_init, c_base)


    def rollout(self, graph, task_goal):
        nb_steps = 0
        _ = self.env.reset(graph, task_goal)
        done = False      
        self.env.to_pomdp()

        self.sim_env.reset(graph, task_goal)
        self.sim_env.to_pomdp()

        gt_actions = ['[walk] <dining_room> (41)', 
                      '[walk] <cupboard> (117)', 
                      '[open] <cupboard> (117)',
                      '[walk] <plate> (2005)',
                      '[grab] <plate> (2005)']#,
                      # '[walk] <table> (226)',
                      # '[putback] <cup> (2009) <table> (226)']

        root_action = None
        root_node = None
        while not done and nb_steps < self.max_episode_length:
            if nb_steps < 2:
                action = gt_actions[nb_steps]
            else:
                action, root_node = sampler(None, root_action, root_node, self.sim_env, self.mcts, nb_steps, None)
                root_action = action
            # action = sampler(None, self.env, self.mcts, nb_steps, None)
            
            reward, state, infos = self.env.step({0: action})
            done = infos['n'][0]['terminate']
            nb_steps += 1
            print(nb_steps, action, reward)

            _, _, _ = self.sim_env.step({0: action})
            print(self.env.get_action_space())

            state = self.env.vh_state.to_dict()
            print([e for e in state['edges'] if 2005 in e.values()])
            print([e for e in state['edges'] if 240 in e.values()])

            obs_graph = self.env.get_observations(0)
            self.bel.update_from_gt_graph(obs_graph)
            new_graph = self.bel.sample_from_belief()
            self.sim_env.reset(new_graph, task_goal)
            sim_state = self.sim_env.vh_state.to_dict()
            print('sim')
            id_goal = 2005
            id_agent = 240
            print([n for n in sim_state['nodes'] if n['category'] == 'Rooms'])
            print([n for n in sim_state['nodes'] if n['id'] == id_goal])
            print([[(n['id'], n['class_name']) for n in sim_state['nodes'] if n['id'] == e['from_id']] for e in sim_state['edges'] if 41 in e.values()])
            print([e for e in sim_state['edges'] if id_goal in e.values() and e['relation_type'] == 'INSIDE'])
            print([e for e in sim_state['edges'] if 117 in e.values() and e['relation_type'] == 'INSIDE'])
            print([e for e in sim_state['edges'] if id_agent in e.values()])

            # print('action_space:', self.env.get_action_space(obj1=['cup', 'cupboard', 'dining_room']))

        # while not done and self.max_episode_length:
        #     if nb_steps < 1:
        #         action = gt_actions[nb_steps]
        #     else:
        #         manager = multiprocessing.Manager()
        #         res = manager.dict()
        #         for start_sample_id in range(0, self.num_samples, self.num_processes):
        #             end_sample_id = min(start_sample_id + self.num_processes, self.num_samples)
        #             jobs = []
        #             for sample_id in range(start_sample_id, end_sample_id):
        #                 p = multiprocessing.Process(target=sampler,
        #                                             args=(sample_id,
        #                                                   self.env,
        #                                                   self.mcts,
        #                                                   nb_steps,
        #                                                   res))
        #                 jobs.append(p)
        #                 p.start()
        #             for p in jobs:
        #                 p.join()
        #         print(res)
        #         tmp_actions = [res[sample_id] for sample_id in range(self.num_samples)]
        #         print(tmp_actions)
        #         if None in tmp_actions:
        #             terminal = True
        #             break
        #         action = max(set(tmp_actions), key=tmp_actions.count)
        #         if terminal: break
        #     if terminal: break
        #         # print('state:', self.env.state)

        #     # action = gt_actions[nb_steps]
        #     # print("|||||||||||||||||||||||||")
        #     # print('edges about character', [x for x in self.env.vh_state.to_dict()['edges'] if x['from_id'] == 65])# and x['relation_type'] in ['INSIDE', 'CLOSE']])
        #     # print('edges about cup', [x for x in self.env.vh_state.to_dict()['edges'] if x['from_id'] == 2009])
        #     # print("|||||||||||||||||||||||||")
        #     reward, state, infos, done = self.env.step(action)
        #     # print(infos)
        #     # print("+++++++++++++++++++++++++")
        #     # print('edges about character', [x for x in self.env.vh_state.to_dict()['edges'] if x['from_id'] == 65])# and x['relation_type'] in ['INSIDE', 'CLOSE']])
        #     # print('edges about cup', [x for x in self.env.vh_state.to_dict()['edges'] if x['from_id'] == 2009])
        #     # print("+++++++++++++++++++++++++")
        #     nb_steps += 1
        #     print(nb_steps, action, reward)
        #     print('action_space:', self.env.get_action_space(obj1=['cup', 'cupboard', 'table']))
