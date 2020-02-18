import pickle
import pdb
import sys
import os
import random
import json


random.seed(10)
home_path = '/Users/shuangli/Desktop/0mit/0research/0icml2020/1virtualhome'
sys.path.append(home_path+'/vh_mdp')
sys.path.append(home_path+'/virtualhome')
sys.path.append(home_path+'/vh_multiagent_models')


import utils
import utils_unity_graph
from simulation.unity_simulator import comm_unity as comm_unity
from simulation.evolving_graph.utils import load_graph_dict
from profilehooks import profile

from interface.envs.envs import UnityEnv
from vh_init import SetInitialGoal


class GetReward:
    def __init__(self, goal, task_name):
        self.goal = goal
        self.task_name = task_name

    def setup_table(self, graph):
        subgoal_reward = 0
        for subgoal in self.goal['setup_table']:
            assert len(subgoal)==1
            subgoal_name = list(subgoal.keys())[0].split('_')
            subgoal_num = list(subgoal.values())[0]
            rel_pos = 'ON'
            obj1 = subgoal_name[1]
            obj2 = int(subgoal_name[3])
            
            obj1_ids = [node['id'] for node in graph['nodes'] if obj1 in node['class_name']]
            obj_on_table_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]

            obj1_on_table_ids = list(set(obj1_ids) & set(obj_on_table_ids))
            
            if len(obj1_on_table_ids)>=subgoal_num:
                subgoal_reward+=1

        if subgoal_reward==len(self.goal['setup_table']):
            return 1
        else:
            return 0


    def clean_table(self,  graph):     
        ## just use table id   
        subgoal = self.goal['clean_table'][0]
        assert len(subgoal)==1
        subgoal_name = list(subgoal.keys())[0].split('_')
        subgoal_num = list(subgoal.values())[0]
        rel_pos = 'ON'
        obj1 = subgoal_name[1]
        obj2 = int(subgoal_name[3])
        
        obj_on_table_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]
        if len(obj_on_table_ids)==0:
            return 1
        else:
            return 0


    def put_dishwasher(self, graph):
        subgoal_reward = 0
        for subgoal in self.goal['put_dishwasher']:
            assert len(subgoal)==1
            subgoal_name = list(subgoal.keys())[0].split('_')
            subgoal_num = list(subgoal.values())[0]
            rel_pos = 'INSIDE'
            obj1 = subgoal_name[1]
            obj2 = int(subgoal_name[3])
            
            obj1_ids = [node['id'] for node in graph['nodes'] if obj1 in node['class_name']]
            obj_inside_dishwasher_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]

            obj1_inside_dishwasher_ids = list(set(obj1_ids) & set(obj_inside_dishwasher_ids))
            if len(obj1_inside_dishwasher_ids)>=subgoal_num:
                subgoal_reward+=1

        if subgoal_reward==len(self.goal['put_dishwasher']):
            return 1
        else:
            return 0


    def unload_dishwasher(self,  graph):
        subgoal = self.goal['unload_dishwasher'][0]
        assert len(subgoal)==1
        subgoal_name = list(subgoal.keys())[0].split('_')
        subgoal_num = list(subgoal.values())[0]
        rel_pos = 'INSIDE'
        obj1 = subgoal_name[1]
        obj2 = int(subgoal_name[3])
        
        obj_inside_dishwasher_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]
        if len(obj_inside_dishwasher_ids)==0:
            return 1
        else:
            return 0



    def put_fridge(self, graph):
        subgoal_reward = 0
        for subgoal in self.goal['put_fridge']:
            assert len(subgoal)==1
            subgoal_name = list(subgoal.keys())[0].split('_')
            subgoal_num = list(subgoal.values())[0]
            rel_pos = 'INSIDE'
            obj1 = subgoal_name[1]
            obj2 = int(subgoal_name[3])
            
            obj1_ids = [node['id'] for node in graph['nodes'] if obj1 in node['class_name']]
            obj_inside_fridge_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]

            obj1_inside_fridge_ids = list(set(obj1_ids) & set(obj_inside_fridge_ids))
            if len(obj1_inside_fridge_ids)>=subgoal_num:
                subgoal_reward+=1

        if subgoal_reward==len(self.goal['put_fridge']):
            return 1
        else:
            return 0



    def read_book(self, graph):
        subgoal = self.goal['read_book']
        return 1


    def prepare_food(self, graph):
        subgoal_reward = 0
        for subgoal in self.goal['prepare_food']:
            assert len(subgoal)==1
            subgoal_name = list(subgoal.keys())[0].split('_')
            subgoal_num = list(subgoal.values())[0]
            rel_pos = 'ON'
            obj1 = subgoal_name[1]
            obj2 = int(subgoal_name[3])
            
            obj1_ids = [node['id'] for node in graph['nodes'] if obj1 in node['class_name']]
            obj_on_table_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]

            obj1_on_table_ids = list(set(obj1_ids) & set(obj_on_table_ids))
            if len(obj1_on_table_ids)>=subgoal_num:
                subgoal_reward+=1

        if subgoal_reward==len(self.goal['prepare_food']):
            return 1
        else:
            return 0

    def watch_tv(self, graph):
        subgoal = self.goal['watch_tv'][0]
        assert len(subgoal)==1
        subgoal_name = list(subgoal)[0].split('_')
        tv_id = int(subgoal_name[0])
        tv_state = subgoal_name[1]
        
        tv_states = [node for node in graph['nodes'] if tv_id==node['id']]
        # print(tv_states, len(tv_states))
        assert len(tv_states)==1
        
        if len(tv_states[0]['states'])==0:
            return 0
        else:
            if tv_states[0]['states'][0]==tv_state:
                return 1
            else:
                return 0


    def setup_table_prepare_food(self, graph):
        reward1 = self.setup_table(graph)
        reward2 = self.prepare_food(graph)
        return reward1 and reward2

    def setup_table_read_book(self, graph):
        reward1 = self.setup_table(graph)
        reward2 = self.read_book(graph, start=False)
        return reward1 and reward2
    
    def setup_table_watch_tv(self, graph):
        reward1 = self.setup_table(graph)
        reward2 = self.watch_tv(graph, start=False)
        return reward1 and reward2

    def setup_table_put_fridge(self, graph):
        reward1 = self.setup_table(graph)
        reward2 = self.put_fridge(graph, start=False)
        return reward1 and reward2

    def setup_table_put_diswasher(self, graph):
        reward1 = self.setup_table(graph)
        reward2 = self.put_diswasher(graph, start=False)
        return reward1 and reward2




if __name__ == "__main__":
    # envs = UnityEnv(num_agents=1)
    # graph = envs.get_graph()

    comm = comm_unity.UnityCommunication()
    comm.reset()
    s, graph = comm.environment_graph()


    success_init_graph = pickle.load( open( "result/init7_50.p", "rb" ) )
    
    for data in success_init_graph:
        apartment = data['apartment']
        task_name = data['task_name']
        init_graph = data['init_graph']
        goal = data['goal']

        comm.reset(apartment-1)
        s, graph = comm.environment_graph()
        success, message = comm.expand_scene(init_graph)
        print(success, message)

        get_reward = GetReward(goal, task_name)
        reward = getattr(get_reward, task_name)(graph)
        print(task_name, reward)
    

    












