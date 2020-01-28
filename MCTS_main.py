import gym
import ipdb
import sys
sys.path.append('../vh_mdp')
sys.path.append('../virtualhome')
import vh_graph
from vh_graph.envs import belief, vh_env
import utils_viz
import utils
import json
import random
import numpy as np
import cProfile
import ipdb
from simulation.evolving_graph.utils import load_graph_dict
from simulation.unity_simulator import comm_unity as comm_unity
from profilehooks import profile
import pickle
sys.argv = ['-f']

from agents import MCTS_agent, PG_agent

import timeit

# Options, should go as argparse arguments
agent_type = 'MCTS' # PG/MCTS
simulator_type = 'unity' # unity/python
dataset_path = '../dataset_toy4/init_envs/'


class UnityEnvWrapper:
    def __init__(self, comm, num_agents):
        self.comm = comm
        self.num_agents = num_agents
        self.graph = None

        comm.reset(0)
        for _ in range(self.num_agents):
            self.comm.add_character()
        
        self.get_graph()
        self.test_prep()

    def get_graph(self):

        _, self.graph = self.comm.environment_graph()
        return self.graph

    def test_prep(self):
        node_id_new = 2007
        s, graph = self.comm.environment_graph()
        table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
        container_id = [node['id'] for node in graph['nodes'] if node['class_name'] in ['fridge', 'freezer']][0]
        drawer_id = [node['id'] for node in graph['nodes'] if node['class_name'] in ['kitchencabinets']][0]


        id2node = {node['id']: node for node in graph['nodes']}

        # plates = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == table_id and id2node[edge['from_id']]['class_name'] == 'plate']
        # graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in plates and edge['to_id'] not in plates]
        # edge_plates = [{'from_id': plate_id, 'to_id': drawer_id, 'relation_type': 'INSIDE'} for plate_id in plates] 
        # graph['edges'] += edge_plates
        #ipdb.set_trace()


        new_node = {'id': node_id_new, 'class_name': 'glass', 'states': [], 'properties': ['GRABBABLE']}
        new_edge = {'from_id': node_id_new, 'relation_type': 'INSIDE', 'to_id': container_id}
        graph['nodes'].append(new_node)
        graph['edges'].append(new_edge)
        success = self.comm.expand_scene(graph)
        print(success)

    def agent_ids(self):
        return [x['id'] for x in self.graph['nodes'] if x['class_name'] == 'character']

    def execute(self, actions): # dictionary from agent to action
        # Get object to interact with

        # This solution only works for 2 agents, we can scale it for more agents later
        
        agent_do = list(actions.keys())
        if len(actions.keys()) > 1:
            objects_interaction = [x.split('(')[1].split(')')[0] for x in actions.values() if 'walk' not in x]
            if len(set(objects_interaction)) == 1:
                agent_do = [random.choice([0,1])]

        script_list = ['']
        for agent_id in agent_do:
            script = actions[agent_id]
            current_script = ['<char{}> {}'.format(agent_id, script)]
            if 'walk' not in script:
                # TODO: very hacky, improve
                if '[put' in script:
                    current_script = ['<char{}> [Find] {})'.format(agent_id, script.split(') ')[1])] + current_script
                else:
                    current_script = ['<char{}> [Find] {}'.format(agent_id, script.split('] ')[1])] + current_script
            
            if len(script_list) < len(current_script):
                script_list.append('')

            script_list = [x+ '|' +y if len(x) > 0 else y for x,y in zip (script_list, current_script)]
            
        # script_all = script_list
        print(script_list)
        success, message = self.comm.render_script(script_list, image_synthesis=[])
        if not success:
            ipdb.set_trace()
        result = {}
        for agent_id in agent_do:
            result[agent_id] = (success, message)
        
        return result

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

def inside_not_trans(graph):
    inside_node = {}
    other_edges = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] not in inside_node:
                inside_node[edge['from_id']] = []
            inside_node[edge['from_id']].append(edge['to_id'])
        else:
            other_edges.append(edge)
    # Make sure we make trasnsitive first
    inside_trans = {}
    def inside_recursive(curr_node_id):
        if curr_node_id in inside_trans:
            return inside_trans[node_id]
        if curr_node_id not in inside_node.keys():
            return []
        else:
            all_parents = []
            for node_id_parent in inside_node[curr_node_id]:
                curr_parents = inside_recursive(node_id_parent)
                all_parents += curr_parents

            if len(all_parents) > 0:
                inside_trans[curr_node_id] = list(set(all_parents))
            return all_parents

    for node_id in inside_node.keys():
        if len(inside_node[node_id]) > 1:
            inside_recursive(node_id)
        else:
            other_edges.append({'from_id':node_id, 'relation_type': 'INSIDE', 'to_id': inside_node[node_id][0]})

    num_parents = {}
    for node in graph['nodes']:
        if node['id'] not in inside_trans.keys():
            num_parents[node['id']] = 0
        else:
            num_parents[node['id']] = len(inside_trans[node['id']])

    edges_inside = []
    for node_id, nodes_inside in inside_trans.items():
        all_num_parents = [num_parents[id_n] for id_n in nodes_inside]
        max_np = max(all_num_parents)
        node_select = [node_inside[i] for i, np in enumerate(all_num_parents) if np == max_np][0]
        edges_inside.append({'from_id':node_id, 'relation_type': 'INSIDE', 'to_id': node_select})
    graph['edges'] = edges_inside + other_edges
    return graph


def interactive_rollout():

    num_agents = 1
    env = vh_env.VhGraphEnv(n_chars=num_agents)
    # env = gym.make('vh_graph-v0')


    comm = comm_unity.UnityCommunication()
    unity_simulator = UnityEnvWrapper(comm, num_agents)    
    agent_ids =  unity_simulator.agent_ids()
    agents = []
    for agent_id in agent_ids:
        agents.append(MCTS_agent(env=env,
                           agent_id=agent_id,
                           max_episode_length=5,
                           num_simulation=100,
                           max_rollout_steps=3,
                           c_init=0.1,
                           c_base=1000000,
                           num_samples=1,
                           num_processes=1))

    # Preparing the goal
    graph = unity_simulator.get_graph()
    glasses_id = [node['id'] for node in graph['nodes'] if 'wineglass' in node['class_name']]
    table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    goals = ['put_{}_{}'.format(glass_id, table_id) for glass_id in glasses_id][:2]
    task_goal = {}

    for i in range(num_agents):
        task_goal[i] = goals
    
    # Assumption: At the beggining the character is not close to anything
    graph = inside_not_trans(graph)
    for i in range(num_agents):
        agents[i].reset(graph, task_goal, seed=i)

    last_position = None
    last_walk_room = False
    num_steps = 0

    print('Starting')
    while True:
        graph = unity_simulator.get_graph()
        if num_steps == 0:
            graph['edges'] = [edge for edge in graph['edges'] if not (edge['relation_type'] == 'CLOSE' and (edge['from_id'] in agent_ids or edge['to_id'] in agent_ids))]

        num_steps += 1
        id2node = {node['id']: node for node in graph['nodes']}
        
        
        print('CHARACTER LOCATION')
        print([edge for edge in graph['edges'] if edge['from_id'] == agent_ids[0] and edge['relation_type'] == 'INSIDE'])

        graph = inside_not_trans(graph)
        # Inside seems to be working now
        
        if last_position is not None:    
            character_close = lambda x, char_id: x['relation_type'] in ['CLOSE'] and (
                (x['from_id'] == char_id or x['to_id'] == char_id))
            character_location = lambda x, char_id: x['relation_type'] in ['INSIDE'] and (
                (x['from_id'] == char_id or x['to_id'] == char_id))
            
            if last_walk_room:
                graph['edges'] = [edge for edge in graph['edges'] if not character_location(edge, agent_id) and not character_close(edge, agent_id)]
            else:
                graph['edges'] = [edge for edge in graph['edges'] if not character_location(edge, agent_id)]
            graph['edges'].append({'from_id': agent_id, 'relation_type': 'INSIDE', 'to_id': last_position})


        env.reset(graph , task_goal)
        

        action_dict = {}
        for i, agent in enumerate(agents):
            agent.sample_belief(env.get_observations(char_index=i))
            agent.sim_env.reset(agent.previous_belief_graph, task_goal)
            action, info = agent.get_action(task_goal[0])
            if action is None:
                print("DONE")
                exit()
            else:
                action_dict[i] = action
                print(action, info['plan'][:3])

        dict_results = unity_simulator.execute(action_dict)
        

        # success, message = comm.render_script(script, image_synthesis=[])
        for char_id, (success, message) in dict_results.items():
            if not success:
                print(char_id, message)


        # last_walk_room = False
        # if success:
        #     if 'walk' in action:
        #         walk_id = int(action.split('(')[1][:-1])
        #         if id2node[walk_id]['category'] == 'Rooms':
        #             last_position = walk_id
        #             last_walk_room = True
        # else:
        #     print(message)


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


