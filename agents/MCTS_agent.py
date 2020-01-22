import numpy as np
from pathlib import Path
import random
import time
import math
import copy
import importlib
import multiprocessing
import ipdb


from vh_graph.envs import belief as Belief
from vh_graph.envs.vh_env import VhGraphEnv

from MCTS import *

def find_heuristic(agent_id, env_graph, observations, object_target):
    id2node = {node['id']: node for node in env_graph['nodes']}
    target = int(object_target.split('_')[-1])
    observation_ids = [x['id'] for x in observations['nodes']]
    

    action_list = []
    while target not in observation_ids:
        containers = [e['to_id'] for e in env_graph['edges']
                      if e['from_id'] == target and e['relation_type'] == 'INSIDE']
        
        # If the object is a room, we have to walk to what is insde
        if id2node[containers[0]]['category'] == 'Rooms':
            action_list = [('walk', (id2node[target]['class_name'], target), None)] + action_list 
       
        elif 'CLOSED' in id2node[containers[0]]['states'] or not 'OPEN' in id2node[containers[0]]['states']:
            action = ('open', (id2node[containers[0]]['class_name'], containers[0]), None)
            action_list = [action] + action_list
        target = containers[0]
    
    ids_character = [x['to_id'] for x in observations['edges'] if
                     x['from_id'] == agent_id and x['relation_type'] == 'CLOSE']

    if target not in ids_character:
        # If character is not next to the object, walk there
        action_list = [('walk', (id2node[target]['class_name'], target), None)]+ action_list

    return action_list


def grab_heuristic(agent_id, env_graph, observations, object_target):
    print(object_target)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if (edge['from_id'] == agent_id and edge['to_id'] == target_id and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if (edge['from_id'] == agent_id and edge['relation_type'] == 'GRABBED')]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('grab', (target_node['class_name'], target_id), None)]
    else:
        target_action = []

    if len(agent_close) > 0 and target_id in observed_ids > 0:
        return target_action
    else:
        return find_heuristic(agent_id, env_graph, observations, object_target)+target_action


def put_heuristic(agent_id, env_graph, observations, target):
    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]
    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]

    grab_obj1 = grab_heuristic(agent_id, env_graph, observations, 'grab_' + str(target_node['id']))
    find_obj2 = find_heuristic(agent_id, env_graph, observations, 'find_' + str(target_node2['id']))

    action = [('putback', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    res = grab_obj1 + find_obj2 + action
    return res

def get_plan(sample_id, root_action, root_node, env, mcts, nb_steps, goal_ids, res):
    init_vh_state = env.vh_state
    init_state = env.state
    observations = env.get_observations(char_index=0)
    # print('init state:', init_state)

    q = goal_ids

    l = 0


    import time
    t1 = time.time()


    if env.is_terminal(0, init_state):
        terminal = True
        if sample_id is not None:
            res[sample_id] = None
        return
    # if root_action is None:
    root_node = Node(id=(root_action, [init_vh_state, init_state, goal_ids, 0, []]),
                     num_visited=0,
                     sum_value=0,
                     is_expanded=False)
    curr_node = root_node
    heuristic_dict = {
        'find': find_heuristic,
        'grab': grab_heuristic,
        'put': put_heuristic
    }
    next_root, plan = mcts.run(curr_node,
                               nb_steps,
                               heuristic_dict)
    print('TS', time.time() - t1)
    print('init state:', [e for e in init_state['edges'] if e['from_id'] == 162])
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
    def __init__(self, env, agent_id,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base,
                 num_samples=1, num_processes=1, comm=None):
        self.env = env
        self.agent_id = agent_id
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

        # Indicates whether there is a unity simulation
        self.comm = comm


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


    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
        edges = [edge for edge in graph['edges'] if edge['from_id'] == char_id]
        print('Character:')
        print(edges)
        print('---')


    def get_action(self, graph, task_goal):
        first_time = time.time()
        self.mcts = MCTS(self.sim_env, self.agent_id, self.max_episode_length,
                         self.num_simulation, self.max_rollout_steps,
                         self.c_init, self.c_base)
        if self.mcts is None:
            raise Exception

        # TODO: is this correct?
        nb_steps = 0
        root_action = None
        root_node = None



        plan, root_node = get_plan(None, root_action, root_node, self.sim_env, self.mcts, nb_steps, task_goal, None)

        action = plan[0]
        info = {
            'plan': plan,
            'action': action,
            'belief': copy.deepcopy(self.belief.edge_belief),
            'belief_graph': copy.deepcopy(self.sim_env.vh_state.to_dict())
        }
        return action, info

    def reset(self, graph, task_goal):
        if self.comm is not None:
            s, graph = self.comm.environment_graph()


        self.env.reset(graph, task_goal)
        self.env.to_pomdp()
        gt_state = self.env.vh_state.to_dict()
        self.belief = Belief.Belief(gt_state, seed=0)
        self.sample_belief(self.env.get_observations(char_index=0))
        self.sim_env.reset(self.previous_belief_graph, task_goal)
        self.sim_env.to_pomdp()



    def rollout(self, graph, task_goal):

        self.reset(graph, task_goal)
        nb_steps = 0
        done = False

        root_action = None
        root_node = None
        obs_graph = None
        # print(self.sim_env.pomdp)


        history = {'belief': [], 'plan': [], 'action': [], 'belief_graph': []}
        while not done and nb_steps < self.max_episode_length:

            action, info = self.get_action(graph, task_goal[0])
            plan, belief, belief_graph = info['plan'], info['belief'], info['belief_graph']

            if obs_graph is not None:
                self.get_relations_char(obs_graph)

            history['belief'].append(belief)
            history['plan'].append(plan)
            history['action'].append(action)
            history['belief_graph'].append(belief_graph)

            reward, state, infos = self.env.step({0: action})
            done = abs(reward[0] - 1.0) < 1e-6
            nb_steps += 1


            obs_graph = self.env.get_observations(char_index=0)
            self.sample_belief(self.env.get_observations(char_index=0))
            self.sim_env.reset(self.previous_belief_graph, task_goal)


            state = self.env.vh_state.to_dict()


            sim_state = self.sim_env.vh_state.to_dict()
            self.sim_env.to_pomdp()


        import pdb
        return history

