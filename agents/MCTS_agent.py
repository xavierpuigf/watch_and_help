import numpy as np
from pathlib import Path
import random
import time
import math
import copy
import importlib
import json
import multiprocessing
import ipdb
from profilehooks import profile


from vh_graph.envs import belief as Belief
from vh_graph.envs.vh_env import VhGraphEnv

from MCTS import *

def find_heuristic(agent_id, char_index, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    id2node = {node['id']: node for node in env_graph['nodes']}
    containerdict = {edge['from_id']: edge['to_id'] for edge in env_graph['edges'] if edge['relation_type'] == 'INSIDE'}
    target = int(object_target.split('_')[-1])
    observation_ids = [x['id'] for x in observations['nodes']]
    try:
        room_char = [edge['to_id'] for edge in env_graph['edges'] if edge['from_id'] == agent_id and edge['relation_type'] == 'INSIDE'][0]
    except:
        ipdb.set_trace()

    action_list = []
    cost_list = []
    while target not in observation_ids:
        try:
            container = containerdict[target]
        except:
            print(id2node[target])
            ipdb.set_trace()
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

def grab_heuristic(agent_id, char_index, env_graph, simulator, object_target):
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
        find_actions, find_costs = find_heuristic(agent_id, char_index, env_graph, simulator, object_target)
        return find_actions + target_action, find_costs + cost

def put_heuristic(agent_id, char_index, env_graph, simulator, target):
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
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, env_graph, simulator, 'grab_' + str(target_node['id']))
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
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, env_graph_new, simulator, 'find_' + str(target_node2['id']))
    action = [('putback', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost = [0.05]
    res = grab_obj1 + find_obj2 + action
    cost_list = cost_grab_obj1 + cost_find_obj2 + cost

    #print(res, target)
    return res, cost_list

def putIn_heuristic(agent_id, char_index, env_graph, simulator, target):
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
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, env_graph, simulator, 'grab_' + str(target_node['id']))
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
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, env_graph_new, simulator, 'find_' + str(target_node2['id']))
    target_put_state = target_node2['states']
    action_open = [('open', (target_node2['class_name'], target_put))]
    action_put = [('putin', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    action_close = [('close', (target_node2['class_name'], target_put))]
    cost_open = [0.05]
    cost_put = [0.05]
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

    print(ids_interaction)

    new_graph = {
            "edges": [edge for edge in state['edges'] if edge['from_id'] in ids_interaction and edge['to_id'] in ids_interaction],
            "nodes": [id2node[id_node] for id_node in ids_interaction]
    }

    return new_graph

def check_progress(state, goal_spec):
    """TODO: add more predicate checkers; currently only ON"""
    unsatisfied = {}
    satisfied = {}
    id2node = {node['id']: node for node in state['nodes']}
    for key, value in goal_spec.items():
        elements = key.split('_')
        unsatisfied[key] = value if elements[0] in ['on', 'inside'] else 0 
        satisfied[key] = [None] * 2
        satisfied[key]
        satisfied[key] = []
        for edge in state['edges']:
            if elements[0] in ['on', 'inside']:
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0] == 'offOn':
                if edge['relation_type'].lower() == 'on' and edge['to_id'] == int(elements[2]) and  (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
            elif elements[1] == 'offIn':
                if edge['relation_type'].lower() == 'in' and edge['to_id'] == int(elements[2]) and  (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
    return satisfied, unsatisfied

def get_plan(sample_id, root_action, root_node, env, mcts, nb_steps, goal_spec, res, last_subgoal, opponent_subgoal=None):
    init_state = env.state
    print('get plan, ')
    # ipdb.set_trace()
    # if mcts.last_opened is not None:
    #     ipdb.set_trace()
    #     for node in init_state['nodes']:
    #          if '()'.format(node['id']) == mcts.last_opened[1]:
    #             if 'CLOSED' in node['states']:
    #                 mcts.last_opened = None
    #                 break
    if True: # clean graph
        init_state = clean_graph(init_state, goal_spec, mcts.last_opened)
        init_vh_state = env.get_vh_state(init_state)
    else:
        init_vh_state = env.vh_state

    satisfied, unsatisfied = check_progress(init_state, goal_spec)
    print('goal_spec:', goal_spec)
    # print('get plan:', init_state)

    if env.is_terminal(0, init_state):
        terminal = True
        if sample_id is not None:
            res[sample_id] = None
        return
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
        'putIn': putIn_heuristic
    }
    next_root, plan, subgoals = mcts.run(curr_node,
                               nb_steps,
                               heuristic_dict,
                               last_subgoal,
                               opponent_subgoal)
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
    def __init__(self, unity_env, agent_id, char_index,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base,
                 num_samples=1, num_processes=1, comm=None):
        self.unity_env = unity_env
        self.env = unity_env.env

        self.logging = False

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

        self.mcts = MCTS(self.sim_env, self.agent_id, self.char_index, self.max_episode_length,
                         self.num_simulation, self.max_rollout_steps,
                         self.c_init, self.c_base)
        if self.mcts is None:
            raise Exception

        # Indicates whether there is a unity simulation
        self.comm = comm


    def sample_belief(self, obs_graph):
        self.belief.update_from_gt_graph(obs_graph)

        # TODO: probably these 2 cases are not needed
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


    @profile
    def get_action(self, task_goal, last_action, last_subgoal, opponent_subgoal):


        """TODO: just close fridge, dishwasher?"""
        
        # if last_action is None:
        #     action = '[walk] <kitchen> (58)'
        #     plan = [action]
        if last_action is not None and last_action.split(' ')[0] == '[putin]': # close the door (may also need to check if it has a door)
            elements = last_action.split(' ')
            action = '[close] {} {}'.format(elements[3], elements[4])
            plan = [action]
            subgoals = [last_subgoal]
        else:
            

            # TODO: is this correct?
            nb_steps = 0
            root_action = None
            root_node = None



            plan, root_node, subgoals = get_plan(None, root_action, root_node, self.sim_env, self.mcts, nb_steps, task_goal, None, last_subgoal, opponent_subgoal)

            if len(plan) > 0:
                action = plan[0]
            else:
                action = None
        info = {
            'plan': plan,
            'action': action,
            'subgoals': subgoals
            # 'belief': copy.deepcopy(self.belief.edge_belief),
            # 'belief_graph': copy.deepcopy(self.sim_env.vh_state.to_dict())
        }
        return action, info

    def reset(self, graph, task_goal, seed=0):
        if self.comm is not None:
            s, graph = self.comm.environment_graph()


        """TODO: do no need this?"""
        self.env.reset(graph, task_goal)
        self.env.to_pomdp()
        gt_state = self.env.vh_state.to_dict()


        self.belief = Belief.Belief(gt_state, agent_id=self.agent_id, seed=seed)
        self.sample_belief(self.env.get_observations(char_index=self.char_index))
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

        unsatisfied = {}
        satisfied = {}
        for key, value in task_goal[0]:
            unsatisfied[key] = value
            satisfied[key] = [None] * 2
            satisfied[key][0] = 0
            satisfied[key][1] = []

        last_action = None

        while not done and nb_steps < self.max_episode_length:

            """TODO: get satisfied, unsatisfied from the latest state"""

            action, info = self.get_action(task_goal[0], last_action)
            last_action = action
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


            obs_graph = self.env.get_observations(char_index=self.char_index)
            self.sample_belief(self.env.get_observations(char_index=self.char_index))
            self.sim_env.reset(self.previous_belief_graph, task_goal)
            self.sim_env.to_pomdp()

            state = self.env.vh_state.to_dict()


            sim_state = self.sim_env.vh_state.to_dict()
            


        import pdb
        return history


    def run(self, single_agent=False):
        graph = self.env.state
        task_goal = self.unity_env.task_goal
        ## --------------------------------------------------------
        # graph = self.unity_env.inside_not_trans(graph)
        all_agent_id = self.unity_env.get_all_agent_id()
        ## --------------------------------------------------------
        num_agents = 1 if single_agent else 2

        if not single_agent:
            self.reset(graph, task_goal, seed=self.agent_id)
        
            
        last_position = [200 for _ in all_agent_id]
        last_walk_room = [False for _ in all_agent_id]
        num_steps = 0


        saved_info = {'action': {0: [], 1: []}, 
                      'plan': {0: [], 1: []},
                      'subgoal': {0: [], 1: []},
                      'init_pos': {0: None, 1: None}}

        last_actions = [None] * 2
        last_subgoals = [None] * 2

        print('Starting')
        while True:
            graph = self.unity_env.get_graph()
            # pdb.set_trace()
            # if num_steps == 0:
            #     graph['edges'] = [edge for edge in graph['edges'] if not (edge['relation_type'] == 'CLOSE' and (edge['from_id'] in all_agent_id or edge['to_id'] in all_agent_id))]
            # graph = self.unity_env.inside_not_trans(graph)


            num_steps += 1
            id2node = {node['id']: node for node in graph['nodes']}
            for agent_id in range(num_agents):
                saved_info['init_pos'][agent_id] = id2node[all_agent_id[agent_id]]['bounding_box']['center']
            
            ##### We won't need this once the character location is working well ####

            # print('INSIDE', [edge for edge in graph['edges'] if edge['from_id'] in all_agent_id and edge['relation_type'] == 'INSIDE'])
            # # Inside seems to be working now
            # for it, agent_id in enumerate(all_agent_id):  
            #     if last_position[it] is not None: 
            #         character_close = lambda x, char_id: x['relation_type'] in ['CLOSE'] and (
            #             (x['from_id'] == char_id or x['to_id'] == char_id))
            #         character_location = lambda x, char_id: x['relation_type'] in ['INSIDE'] and (
            #             (x['from_id'] == char_id or x['to_id'] == char_id))
                    
            #         if last_walk_room[it]:
            #             graph['edges'] = [edge for edge in graph['edges'] if not character_location(edge, agent_id) and not character_close(edge, agent_id)]
            #         else:
            #             graph['edges'] = [edge for edge in graph['edges'] if not character_location(edge, agent_id)]
            #         graph['edges'].append({'from_id': agent_id, 'relation_type': 'INSIDE', 'to_id': last_position[it]})


            # self.unity_env.env.reset(graph, task_goal)

            # print([edge for edge in graph['edges'] if edge['relation_type'] == 'CLOSE' and (edge['from_id'] == 1 or edge['from_id'] == 299 or edge['to_id'] == 299 or edge['to_id'] == 1)])
            # print([edge for edge in graph['edges'] if edge['relation_type'] == 'INSIDE' and (edge['from_id'] == 1 or edge['from_id'] == 299 or edge['to_id'] == 299 or edge['to_id'] == 1)])
            # print('unity env graph:', [edge for edge in graph['edges'] if edge['from_id'] == 306 or edge['to_id'] == 306])
            # print('unity env graph:', [edge for edge in graph['edges'] if edge['from_id'] == 202 or edge['to_id'] == 202])
            print('unity env graph:', [edge for edge in graph['edges'] if edge['from_id'] in all_agent_id or edge['to_id'] in all_agent_id])
            print('unity env graph:', [edge for edge in graph['edges'] if edge['from_id'] == 213 or edge['to_id'] == 213])
            ##########


            ## --------------------------------------------------------
            system_agent_action, system_agent_info = self.unity_env.get_system_agent_action(task_goal, last_actions[0], last_subgoals[0])
            ## --------------------------------------------------------

            last_actions[0] = system_agent_action
            last_subgoals[0] = system_agent_info['subgoals'][0] if len(system_agent_info['subgoals']) > 0 else None

            print(system_agent_info['plan'][:3])
            saved_info['action'][0].append(system_agent_action)
            saved_info['plan'][0].append(system_agent_info['plan'][:3])
            saved_info['subgoal'][0].append(system_agent_info['subgoals'][:2])
            print('Alice action:', system_agent_action)

            action_dict = {}
            if system_agent_action is not None:
                action_dict[0] = system_agent_action
            if single_agent:
                my_agent_action = None
            else:
                observations = self.env.get_observations(char_index=1)
                self.sample_belief(observations)
                self.sim_env.reset(self.previous_belief_graph, task_goal)
                my_agent_action, my_agent_info = self.get_action(task_goal[1], last_actions[1], last_subgoals[1], last_subgoals[0])

                last_actions[1] = my_agent_action
                print('bob subgoal:', my_agent_info['subgoals'])
                last_subgoals[1] = my_agent_info['subgoals'][0] if len(my_agent_info['subgoals']) > 0 else None

                if my_agent_action is None:
                    print("system my action is None! DONE!")
                    # ipdb.set_trace()
                else:
                    action_dict[1] = my_agent_action
                
                print(my_agent_info['plan'][:3])

                saved_info['action'][1].append(my_agent_action)
                saved_info['plan'][1].append(my_agent_info['plan'][:3])
                saved_info['subgoal'][1].append(my_agent_info['subgoals'][:2])
            ## --------------------------------------------------------
            #self.unity_env.print_action(system_agent_action, my_agent_action)
            # infos = self.unity_env.unity_simulator.execute(action_dict)
            obs, reward, done, infos = self.unity_env.step_2agents_python(action_dict)
            print('done:', done)
            # obs, reward, done, infos = self.unity_env.step_with_system_agent_oracle(my_agent_action)
            ## --------------------------------------------------------

            # for char_id, (success, message) in infos.items():
            #     if not success:
            #         print(char_id, message)



            # if success:
            #     for it, agent_id in enumerate(all_agent_id):
                    
            #         last_walk_room[it] = False
            #         if it in action_dict:
            #             action = action_dict[it]
            #         else:
            #             action = None
            #         if action is not None and 'walk' in action:
            #             walk_id = int(action.split('(')[1][:-1])
            #             if id2node[walk_id]['category'] == 'Rooms':
            #                 last_position[it] = walk_id
            #                 last_walk_room[it] = True

            if self.logging:
                with open('../logs/logs_agent.json', 'w+') as f:
                    f.write(json.dumps(saved_info, indent=4))

            # obs, reward, done, infos = self.unity_env.step_alice()
            if done[0]: # ended
                break

