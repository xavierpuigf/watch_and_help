import numpy as np
import pdb
import random

import vh_graph
from vh_graph.envs import belief, vh_env
from simulation.unity_simulator import comm_unity as comm_unity

from agents import MCTS_agent, PG_agent


class UnityEnvWrapper:
    def __init__(self, comm, num_agents):
        self.comm = comm
        self.num_agents = num_agents
        self.graph = None

        comm.reset(0)
        characters = ['Chars/Male1', 'Chars/Female1']
        for i in range(self.num_agents):
            self.comm.add_character(characters[i])
        
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
        #self.comm.render_script(['<char0> [walk] <livingroom> (319)'], image_synthesis=[]).set_trace()


        new_node = {'id': node_id_new, 'class_name': 'glass', 'states': [], 'properties': ['GRABBABLE']}
        new_edge = {'from_id': node_id_new, 'relation_type': 'INSIDE', 'to_id': container_id}
        graph['nodes'].append(new_node)
        graph['edges'].append(new_edge)
        success = self.comm.expand_scene(graph)
        print(success)

    def agent_ids(self):
        return sorted([x['id'] for x in self.graph['nodes'] if x['class_name'] == 'character'])

    def execute(self, actions): # dictionary from agent to action
        # Get object to interact with

        # This solution only works for 2 agents, we can scale it for more agents later
        
        agent_do = list(actions.keys())
        if len(actions.keys()) > 1:
            if sum([1 for action in actions.values() if 'walk' in action]) == 0:
                objects_interaction = [x.split('(')[1].split(')')[0] for x in actions.values()]
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
        success, message = self.comm.render_script(script_list, image_synthesis=[])
        
        if not success:
            ipdb.set_trace()
        result = {}
        for agent_id in agent_do:
            result[agent_id] = (success, message)
        
        return result




class UnityEnv:
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.env = vh_env.VhGraphEnv(n_chars=self.num_agents)

        comm = comm_unity.UnityCommunication()
        self.unity_simulator = UnityEnvWrapper(comm, self.num_agents)    
        self.agent_ids =  self.unity_simulator.agent_ids()
        self.agents = {}
        

        self.system_agent_id = self.agent_ids[0]

        if self.num_agents>1:
            self.my_agent_id = self.agent_ids[1]

        self.add_system_agent()

        self.actions = {}
        self.actions['system_agent'] = []
        self.actions['my_agent'] = []


    def add_system_agent(self):
        ## Alice model
        self.agents[self.system_agent_id] = MCTS_agent(unity_env=self,
                               agent_id=self.system_agent_id,
                               char_index=0,
                               max_episode_length=5,
                               num_simulation=100,
                               max_rollout_steps=3,
                               c_init=0.1,
                               c_base=1000000,
                               num_samples=1,
                               num_processes=1)

    def get_system_agent_action(self, task_goal):
        self.agents[self.system_agent_id].sample_belief(self.env.get_observations(char_index=0))
        self.agents[self.system_agent_id].sim_env.reset(self.agents[self.system_agent_id].previous_belief_graph, task_goal)
        action, info = self.agents[self.system_agent_id].get_action(task_goal[0])
        
        if action is None:
            print("system agent action is None! DONE!")
            pdb.set_trace()
        # else:
        #     print(action, info['plan'])

        return action, info

    def get_all_agent_id(self):
        return self.agent_ids

    def get_my_agent_id(self):
        if self.num_agents==1:
            error("you haven't set your agent")
        return self.my_agent_id

    def reset(self, graph, task_goal):
        # Assumption: At the beggining the character is not close to anything
        self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)


    def inside_not_trans(self, graph):
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


    def print_action(self, system_agent_action, my_agent_action):
        self.actions['system_agent'].append(system_agent_action)
        self.actions['my_agent'].append(my_agent_action)

        system_agent_actions = self.actions['system_agent']
        my_agent_actions = self.actions['my_agent']
        num_steps = len(system_agent_actions)

        print('**************************************************************************')
        if self.num_agents>1:
            for i in range(num_steps):
                print('step %04d:\t|"system": %s \t\t\t\t\t\t |"my_agent": %s' % (i+1, system_agent_actions[i].ljust(30), my_agent_actions[i]))
        else:
            for i in range(num_steps):
                print('step %04d:\t|"system": %s' % (i+1, system_agent_actions[i]))

        print('**************************************************************************')
        





