import numpy as np
import random
import time
import math
import copy
import importlib
import json
import multiprocessing
import ipdb
import pickle


from . import belief
from envs.graph_env import VhGraphEnv
#
from MCTS import *

import sys
sys.path.append('..')
from utils import utils_environment as utils_env


def find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    id2node = {node['id']: node for node in env_graph['nodes']}
    containerdict = {edge['from_id']: edge['to_id'] for edge in env_graph['edges'] if edge['relation_type'] == 'INSIDE'}
    target = int(object_target.split('_')[-1])
    observation_ids = [x['id'] for x in observations['nodes']]
    try:
        room_char = [edge['to_id'] for edge in env_graph['edges'] if edge['from_id'] == agent_id and edge['relation_type'] == 'INSIDE'][0]
    except:
        print('Error')
        #ipdb.set_trace()

    action_list = []
    cost_list = []
    # if target == 478:
    #     ipdb.set_trace()
    while target not in observation_ids:
        try:
            container = containerdict[target]
        except:
            print(id2node[target])
            #ipdb.set_trace()
        # If the object is a room, we have to walk to what is insde

        if id2node[container]['category'] == 'Rooms':
            action_list = [('walk', (id2node[target]['class_name'], target), None)] + action_list 
            cost_list = [0.5] + cost_list
        
        elif 'CLOSED' in id2node[container]['states'] or ('OPEN' not in id2node[container]['states']):
            action = ('open', (id2node[container]['class_name'], container), None)
            action_list = [action] + action_list
            cost_list = [0.05] + cost_list

        target = container
    
    ids_character = [x['to_id'] for x in observations['edges'] if
                     x['from_id'] == agent_id and x['relation_type'] == 'CLOSE'] + \
                    [x['from_id'] for x in observations['edges'] if
                     x['to_id'] == agent_id and x['relation_type'] == 'CLOSE']

    if target not in ids_character:
        # If character is not next to the object, walk there
        action_list = [('walk', (id2node[target]['class_name'], target), None)]+ action_list
        cost_list = [1] + cost_list

    return action_list, cost_list

def grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if ((edge['from_id'] == agent_id and edge['to_id'] == target_id) or (edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if (edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('grab', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target)
        return find_actions + target_action, find_costs + cost

def turnOn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if ((edge['from_id'] == agent_id and edge['to_id'] == target_id) or (edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if (edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('switchon', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target)
        return find_actions + target_action, find_costs + cost

def sit_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if ((edge['from_id'] == agent_id and edge['to_id'] == target_id) or (edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    on_ids = [edge['to_id'] for edge in env_graph['edges'] if (edge['from_id'] == agent_id and 'ON' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in on_ids:
        target_action = [('sit', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target)
        return find_actions + target_action, find_costs + cost

def put_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]

    if sum([1 for edge in observations['edges'] if edge['from_id'] == target_grab and edge['to_id'] == target_put and edge['relation_type'] == 'ON']) > 0:
        # Object has been placed
        return [], []

    if sum([1 for edge in observations['edges'] if edge['to_id'] == target_grab and edge['from_id'] != agent_id and 'HOLD' in edge['relation_type']]) > 0:
        # Object has been placed
        return None, None

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    id2node = {node['id']: node for node in env_graph['nodes']}
    target_grabbed = len([edge for edge in env_graph['edges'] if edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'] and edge['to_id'] == target_grab]) > 0


    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, 'grab_' + str(target_node['id']))
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == 'walk':
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]['category'] == 'Rooms':
                    object_diff_room = id_room
        
        env_graph_new = copy.deepcopy(env_graph)
        
        if object_diff_room:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if edge['to_id'] != agent_id and edge['from_id'] != agent_id]
            env_graph_new['edges'].append({'from_id': agent_id, 'to_id': object_diff_room, 'relation_type': 'INSIDE'})
        
        else:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if (edge['to_id'] != agent_id and edge['from_id'] != agent_id) or edge['relation_type'] == 'INSIDE']
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, unsatisfied, env_graph_new, simulator, 'find_' + str(target_node2['id']))
    action = [('putback', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost = [0.05]
    res = grab_obj1 + find_obj2 + action
    cost_list = cost_grab_obj1 + cost_find_obj2 + cost

    #print(res, target)
    return res, cost_list

def putIn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]

    if sum([1 for edge in observations['edges'] if edge['from_id'] == target_grab and edge['to_id'] == target_put and edge['relation_type'] == 'ON']) > 0:
        # Object has been placed
        return [], []

    if sum([1 for edge in observations['edges'] if edge['to_id'] == target_grab and edge['from_id'] != agent_id and 'HOLD' in edge['relation_type']]) > 0:
        # Object has been placed
        return None, None

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    id2node = {node['id']: node for node in env_graph['nodes']}
    target_grabbed = len([edge for edge in env_graph['edges'] if edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'] and edge['to_id'] == target_grab]) > 0


    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, 'grab_' + str(target_node['id']))
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == 'walk':
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]['category'] == 'Rooms':
                    object_diff_room = id_room
        
        env_graph_new = copy.deepcopy(env_graph)
        
        if object_diff_room:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if edge['to_id'] != agent_id and edge['from_id'] != agent_id]
            env_graph_new['edges'].append({'from_id': agent_id, 'to_id': object_diff_room, 'relation_type': 'INSIDE'})
        
        else:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if (edge['to_id'] != agent_id and edge['from_id'] != agent_id) or edge['relation_type'] == 'INSIDE']
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, unsatisfied, env_graph_new, simulator, 'find_' + str(target_node2['id']))
    target_put_state = target_node2['states']
    action_open = [('open', (target_node2['class_name'], target_put))]
    action_put = [('putin', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost_open = [0.05]
    cost_put = [0.05]
    

    remained_to_put = 0
    for predicate, count in unsatisfied.items():
        if predicate.startswith('inside'):
            remained_to_put += count
    if remained_to_put == 1: # or agent_id > 1:
        action_close= []
        cost_close = []
    else:
        action_close = [('close', (target_node2['class_name'], target_put))]
        cost_close = [0.05]

    if 'CLOSED' in target_put_state or 'OPEN' not in target_put_state:
        res = grab_obj1 + find_obj2 + action_open + action_put + action_close
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost_open + cost_put + cost_close
    else:
        res = grab_obj1 + find_obj2 + action_put + action_close
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost_put + cost_close

    #print(res, target)
    return res, cost_list

def clean_graph(state, goal_spec, last_opened):
    new_graph = {}
    # get all ids
    ids_interaction = []
    nodes_missing = []
    for predicate in goal_spec:
        elements = predicate.split('_')
        nodes_missing += [int(x) for x in elements if x.isdigit()]
        for x in elements[1:]:
            if x.isdigit():
                nodes_missing += [int(x)]
            else:
                nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == x]
    nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == 'character' or node['category'] in ['Rooms', 'Doors']]

    id2node = {node['id']: node for node in state['nodes']}
    # print([node for node in state['nodes'] if node['class_name'] == 'kitchentable'])
    # print(id2node[235])
    # ipdb.set_trace()
    inside = {}
    for edge in state['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] not in inside.keys():
                inside[edge['from_id']] = []
            inside[edge['from_id']].append(edge['to_id'])
    
    while (len(nodes_missing) > 0):
        new_nodes_missing = []
        for node_missing in nodes_missing:
            if node_missing in inside:
                new_nodes_missing += [node_in for node_in in inside[node_missing] if node_in not in ids_interaction]
            ids_interaction.append(node_missing)
        nodes_missing = list(set(new_nodes_missing))

    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    # for clean up tasks, add places to put objects to
    augmented_class_names = []
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['dishwasher', 'kitchentable']:
                augmented_class_names += ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']
                break
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['sofa', 'chair']:
                augmented_class_names += ['coffeetable']
                break
    containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] in augmented_class_names]
    for obj_id in containers:
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)


    new_graph = {
            "edges": [edge for edge in state['edges'] if edge['from_id'] in ids_interaction and edge['to_id'] in ids_interaction],
            "nodes": [id2node[id_node] for id_node in ids_interaction]
    }

    return new_graph


def get_plan(sample_id, root_action, root_node, env, mcts, nb_steps, goal_spec, res, last_subgoal, last_action, opponent_subgoal=None, verbose=True):
    init_state = env.state

    if True: # clean graph
        init_state = clean_graph(init_state, goal_spec, mcts.last_opened)
        init_vh_state = env.get_vh_state(init_state)
    else:
        init_vh_state = env.vh_state

    satisfied, unsatisfied = utils_env.check_progress(init_state, goal_spec)

    remained_to_put = 0
    for predicate, count in unsatisfied.items():
        if predicate.startswith('inside'):
            remained_to_put += count

    if last_action is not None and last_action.split(' ')[0] == '[putin]' and remained_to_put == 0: # close the door (may also need to check if it has a door)
            elements = last_action.split(' ')
            action = '[close] {} {}'.format(elements[3], elements[4])
            plan = [action]
            subgoals = [last_subgoal]

    # if root_action is None:
    root_node = Node(id=(root_action, [init_vh_state, init_state, goal_spec, satisfied, unsatisfied, 0, []]),
                     num_visited=0,
                     sum_value=0,
                     is_expanded=False)
    curr_node = root_node
    heuristic_dict = {
        'find': find_heuristic,
        'grab': grab_heuristic,
        'put': put_heuristic,
        'putIn': putIn_heuristic,
        'sit': sit_heuristic,
        'turnOn': turnOn_heuristic
    }
    next_root, plan, subgoals = mcts.run(curr_node,
                               nb_steps,
                               heuristic_dict,
                               last_subgoal,
                               opponent_subgoal)
    if verbose:
        print('plan', plan)
        print('subgoal', subgoals)
    if sample_id is not None:
        res[sample_id] = plan
    else:
        return plan, next_root, subgoals


class MCTS_agent:
    """
    MCTS for a single agent
    """
    def __init__(self, agent_id, char_index,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, recursive=False,
                 num_samples=1, num_processes=1, comm=None, logging=False, logging_graphs=False, seed=None):
        self.agent_type = 'MCTS'
        self.verbose = False
        self.recursive = recursive

        #self.env = unity_env.env
        if seed is None:
            seed = random.randint(0,100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.agent_id = agent_id
        self.char_index = char_index
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
        self.verbose = False

        self.mcts = MCTS(self.sim_env, self.agent_id, self.char_index, self.max_episode_length,
                         self.num_simulation, self.max_rollout_steps,
                         self.c_init, self.c_base)

        if self.mcts is None:
            raise Exception

        # Indicates whether there is a unity simulation
        self.comm = comm


    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph['edges']:
            key = (edge['from_id'], edge['to_id'])
            if key not in edge_dict:
                edge_dict[key] = [edge['relation_type']]
                new_edges.append(edge)
            else:
                if edge['relation_type'] not in edge_dict[key]:
                    edge_dict[key] += [edge['relation_type']]
                    new_edges.append(edge)

        graph['edges'] = new_edges
        return graph


    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
        edges = [edge for edge in graph['edges'] if edge['from_id'] == char_id]
        print('Character:')
        print(edges)
        print('---')


    def get_action(self, obs, goal_spec, opponent_subgoal=None):

        self.sample_belief(obs)
        self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})

        last_action = self.last_action
        last_subgoal = self.last_subgoal
            

        # TODO: is this correct?
        nb_steps = 0
        root_action = None
        root_node = None
        verbose = self.verbose


        plan, root_node, subgoals = get_plan(None, root_action, root_node, self.sim_env, self.mcts, nb_steps, goal_spec, None, last_subgoal, last_action, opponent_subgoal, verbose=verbose)
        # ipdb.set_trace()
        if len(plan) > 0:
            action = plan[0]
            action = action.replace('[walk]', '[walktowards]')
        else:
            action = None
        if self.logging:
            info = {
                'plan': plan,
                'subgoals': subgoals,
                'belief': copy.deepcopy(self.belief.edge_belief),
                'belief_graph': copy.deepcopy(self.sim_env.vh_state.to_dict())
            }
            if self.logging_graphs:
                info.update(
                    {'obs': obs['nodes']})
        else:
            info = {}

        self.last_action = action
        self.last_subgoal = subgoals[0] if len(subgoals) > 0 else None
        return action, info

    def reset(self, observed_graph, gt_graph, task_goal, seed=0, simulator_type='python', is_alice=False):

        self.last_action = None
        self.last_subgoal = None
        """TODO: do no need this?"""

        self.previous_belief_graph = None
        self.belief = belief.Belief(gt_graph, agent_id=self.agent_id, seed=seed)
        # print("set")
        self.belief.sample_from_belief()
        graph_belief = self.sample_belief(observed_graph) #self.env.get_observations(char_index=self.char_index))
        try:
            self.sim_env.reset(graph_belief, task_goal)
        except:
            import ipdb

            ipdb.set_trace()
        self.sim_env.to_pomdp()
        self.mcts = MCTS(self.sim_env, self.agent_id, self.char_index, self.max_episode_length,
                         self.num_simulation, self.max_rollout_steps,
                         self.c_init, self.c_base, seed=seed)

