import numpy as np
from pathlib import Path
import random
import time
import math
import copy
import pickle
import importlib
import multiprocessing

import vh_graph
from vh_graph.envs import belief
from vh_graph.envs.vh_env import VhGraphEnv

from MCTS import *


def get_plan(sample_id, root_action, root_node, env, mcts, nb_steps, goal_id, res):
    init_vh_state = env.vh_state
    init_state = env.state
    # print('init state:', init_state)

    q = [goal_id]
    print('INIT', q)

    l = 0
    while l < len(q):
        node_id = q[l]
        l += 1
        q += [e['to_id'] for e in init_state['edges'] \
            if e['from_id'] == node_id and e['relation_type'] == 'INSIDE' \
                and e['to_id'] not in q]
    print(q)
    nodes = [node for node in init_state['nodes'] if node['id'] in q]
    print('init state:', [e for e in init_state['edges'] if e['from_id'] == 162])
    
    print(q, nodes)
    # print('init action space:', env.get_action_space(init_vh_state))
    # print('init action space:', env.get_action_space(init_vh_state, obj1=nodes))
    action_space = []
    for obj in nodes:
        for action in ['walk', 'open']:
            action_space += env.get_action_space(init_vh_state, obj1=obj, action=action)

    print('init action space:', action_space)
    # input('press any key ton continue...')
    if env.is_terminal(0, init_state):
        terminal = True
        res[sample_id] = None
        return
    # if root_action is None:
    root_node = Node(id={root_action: [init_vh_state, init_state]},
                    num_visited=0,
                    sum_value=0,
                    is_expanded=False)
    curr_node = root_node
    next_root, plan  = mcts.run(curr_node, 
                                nb_steps, 
                                nodes,
                                ['walk', 'open'])
    print('plan:', plan)
    # else:
    #     action, _, next_root = mcts.select_next_root(root_node)
    print(root_node.sum_value)
    # print(sample_id, res[sample_id])
    if sample_id is not None:
        res[sample_id] = plan
    else:
        return plan, next_root


class MCTS_agent:
    """
    MCTS for a single agent
    """
    def __init__(self, env, max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, num_samples=1, num_processes=1):
        self.env = env
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes

        self.previous_belief_graph = None


    def sample_belief(self, obs_graph):
        self.belief.update_from_gt_graph(obs_graph)
        if self.previous_belief_graph is None:
            self.belief.reset_belief()
            new_graph = self.belief.sample_from_belief()
            new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
            self.previous_belief_graph = new_graph
        else:
            new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
            self.previous_belief_graph = new_graph


    def rollout(self, graph, task_goal):
        nb_steps = 0
        id_goal = int(task_goal[0].split('_')[-1])
        _ = self.env.reset(graph, task_goal)
        done = False      
        self.env.to_pomdp()
        gt_state = self.env.vh_state.to_dict()
        self.belief = belief.Belief(gt_state)
        self.sample_belief(self.env.get_observations(0))
        self.sim_env.reset(self.previous_belief_graph, task_goal)
        self.sim_env.to_pomdp()

        # # self.sim_env.reset(graph, task_goal)
        # # self.sim_env.to_pomdp()
        # obs_graph = self.env.get_observations(0)
        # new_graph = self.bel.update_graph_from_gt_graph(self.sim_env.state, obs_graph)
        # # new_graph = self.bel.sample_from_belief()
        # self.sim_env.reset(new_graph, task_goal)
        # self.sim_env.to_pomdp()

        gt_actions = ['[walk] <dining_room> (163)',
                      '[walk] <home_office> (246)', 
                      '[walk] <dresser> (284)', 
                      '[open] <dresser> (284)',
                      '[walk] <phone> (2038)']#,
                      # '[grab] <plate> (2005)']#,
                      # '[walk] <table> (226)',
                      # '[putback] <cup> (2009) <table> (226)']

        root_action = None
        root_node = None
        # print(self.sim_env.pomdp)
        history = {'belief': [], 'plan': [], 'action': [], 'sampled_state': []}
        while not done and nb_steps < self.max_episode_length:
            if nb_steps < 0:
                action = gt_actions[nb_steps]
                plan = [action]
            else:
                self.mcts = MCTS(self.sim_env, self.max_episode_length, self.num_simulation, self.max_rollout_steps, self.c_init, self.c_base)
                plan, root_node = get_plan(None, root_action, root_node, self.sim_env, self.mcts, nb_steps, id_goal, None)
                action = plan[0]
                root_action = None#action


            # action = sampler(None, self.env, self.mcts, nb_steps, None)
            
            for action in plan:
                action_space = self.env.get_action_space()
                print('tentative action:', action)
                # print('current action space:', action_space)
                if action in self.env.get_action_space():

                    history['belief'].append(copy.deepcopy(self.belief.edge_belief))
                    history['plan'].append(plan)
                    history['action'].append(action)

                    reward, state, infos = self.env.step({0: action})
                    done = abs(reward[0] - 1.0) < 1e-6
                    _, _, _ = self.sim_env.step({0: action})
                    nb_steps += 1
                    print(nb_steps, action, reward)
                    obs_graph = self.env.get_observations(0)
                    self.sample_belief(self.env.get_observations(0))
                    # self.sim_env.reset_graph(self.previous_belief_graph)
                    self.sim_env.reset(self.previous_belief_graph, task_goal)
                else:
                    break
            
            # goal_string = task_goal
            # goal_id = int(goal_string.split('_')[-1])
            # edges_goal = [x for x in self.env.vh_state.to_dict()['edges']
            #               if x['relation_type'] == 'CLOSE' and x['from_id'] == goal_id and x['to_id'] == self.node_id_char]
            # edge_found = len(edges_goal) > 0
            # goal_id in self.env.observable_object_ids_n[0] and edge_found
            # done = goal_id in self.env.observable_object_ids_n[0] and edge_found
            # reward = int(done)

            # done = infos['n'][0]['terminate']
            

            # _, _, _ = self.sim_env.step({0: action})
            # print(self.env.get_action_space())

            state = self.env.vh_state.to_dict()

            # # if action == plan[-1]:
            # obs_graph = self.env.get_observations(0)
            # self.sample_belief(self.env.get_observations(0))
            # # self.sim_env.reset_graph(self.previous_belief_graph)
            # self.sim_env.reset(self.previous_belief_graph, task_goal)
            # # # new_graph = self.bel.update_graph_from_gt_graph(obs_graph)
            # # self.bel.update_from_gt_graph(obs_graph)
            # # new_graph = self.bel.sample_from_belief()
            # # self.sim_env.reset(new_graph, task_goal)
            sim_state = self.sim_env.vh_state.to_dict()
            self.sim_env.to_pomdp()
            # self.sim_env.vh_state._script_objects = dict(self.env.vh_state._script_objects)
            # print('sim')
            #id_goal = 2038
            id_agent = 162
            # print([n for n in sim_state['nodes'] if n['category'] == 'Rooms'])
            # print([n for n in sim_state['nodes'] if n['id'] == id_goal])
            # print([[(n['id'], n['class_name']) for n in sim_state['nodes'] if n['id'] == e['from_id']] for e in sim_state['edges'] if 41 in e.values()])
            print('real state:', [e for e in state['edges'] if id_goal in e.values()])
            print('real state:', [e for e in state['edges'] if id_agent in e.values()])

            print('sim state:', [e for e in sim_state['edges'] if id_goal in e.values()])# and e['relation_type'] == 'INSIDE'])
            print('sim state:', [e for e in sim_state['edges'] if e['from_id'] == 229])
            # print([e for e in sim_state['edges'] if 117 in e.values() and e['relation_type'] == 'INSIDE'])
            print('sim state:', [e for e in sim_state['edges'] if id_agent in e.values()])
            input('press any key to continue...')

            pickle.dump(history, open('logdir/history.pik', 'wb'))

            # print('action_space:', self.env.get_action_space(obj1=['cup', 'cupboard', 'dining_room']))
