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
from simulation.evolving_graph.utils import load_graph_dict
from profilehooks import profile

from interface.envs.envs import UnityEnv
from vh_init import SetInitialGoal

obj_position = {
    "INSIDE": ["toilet", "bathroom_cabinet", "kitchencabinets", "bathroom_counter", "kitchencounterdrawer", "cabinet", "fridge", "oven", "dishwasher", "microwave"],
    "ON": ["bathroomcabinet", "bathroomcounter", "bed", "bench", "bookshelf", "cabinet", "chair", "coffeetable", "desk", "floor", "fryingpan", "kitchencabinets", "kitchencounter", "kitchentable", "mousemat", "nightstand", "oventray", "plate", "radio", "rug", "sofa", "stove", "towelrack"]
    }
# 'BETWEEN', 'CLOSE', 'FACING', 'INSIDE', 'ON'

class GetReward:
    def __init__(self, goal):
        self.goal = goal

    def setup_table(self, graph):
        subgoal_reward = 0
        for subgoal in self.goal:
            assert len(subgoal)==1
            subgoal_name = list(subgoal.keys())[0].split('_')
            subgoal_num = list(subgoal.values())[0]
            rel_pos = subgoal_name[0]
            obj1 = subgoal_name[1]
            obj2 = int(subgoal_name[2])
            
            obj1_ids = [node['id'] for node in graph['nodes'] if obj1 in node['class_name']]
            obj_on_table_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]

            obj1_on_table_ids = list(set(obj1_ids) & set(obj_on_table_ids))
            if obj1_on_table_ids>=subgoal_num:
                subgoal_reward+=1

        if subgoal_reward==len(self.goal):
            return 1
        else:
            return 0


    def clean_table(self,  graph):
        subgoal = self.goal[0]
        assert len(subgoal)==1
        subgoal_name = list(subgoal.keys())[0].split('_')
        subgoal_num = list(subgoal.values())[0]
        rel_pos = subgoal_name[0]
        obj1 = subgoal_name[1]
        obj2 = int(subgoal_name[2])
        
        obj_on_table_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]
        if len(obj_on_table_ids)==0:
            return 1
        else:
            return 0


    def put_diswasher(self, graph):
        subgoal_reward = 0
        for subgoal in self.goal:
            assert len(subgoal)==1
            subgoal_name = list(subgoal.keys())[0].split('_')
            subgoal_num = list(subgoal.values())[0]
            rel_pos = subgoal_name[0]
            obj1 = subgoal_name[1]
            obj2 = int(subgoal_name[2])
            
            obj1_ids = [node['id'] for node in graph['nodes'] if obj1 in node['class_name']]
            obj_inside_dishwasher_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]

            obj1_inside_dishwasher_ids = list(set(obj1_ids) & set(obj_inside_dishwasher_ids))
            if obj1_inside_dishwasher_ids>=subgoal_num:
                subgoal_reward+=1

        if subgoal_reward==len(self.goal):
            return 1
        else:
            return 0


    def unload_diswasher(self,  graph):
        subgoal = self.goal[0]
        assert len(subgoal)==1
        subgoal_name = list(subgoal.keys())[0].split('_')
        subgoal_num = list(subgoal.values())[0]
        rel_pos = subgoal_name[0]
        obj1 = subgoal_name[1]
        obj2 = int(subgoal_name[2])
        
        obj_inside_dishwasher_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]
        if len(obj_inside_dishwasher_ids)==0:
            return 1
        else:
            return 0



    def put_fridge(self, graph):
        subgoal_reward = 0
        for subgoal in self.goal:
            assert len(subgoal)==1
            subgoal_name = list(subgoal.keys())[0].split('_')
            subgoal_num = list(subgoal.values())[0]
            rel_pos = subgoal_name[0]
            obj1 = subgoal_name[1]
            obj2 = int(subgoal_name[2])
            
            obj1_ids = [node['id'] for node in graph['nodes'] if obj1 in node['class_name']]
            obj_inside_fridge_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]

            obj1_inside_fridge_ids = list(set(obj1_ids) & set(obj_inside_fridge_ids))
            if obj1_inside_fridge_ids>=subgoal_num:
                subgoal_reward+=1

        if subgoal_reward==len(self.goal):
            return 1
        else:
            return 0



    def read_book(self, graph):
        agent_id = [node['id'] for node in graph['nodes'] if 'person' in node['class_name']]
        env_goal = [{'grab_{}'.format(book_id)}]


    def prepare_food(self, graph):
        subgoal_reward = 0
        for subgoal in self.goal:
            assert len(subgoal)==1
            subgoal_name = list(subgoal.keys())[0].split('_')
            subgoal_num = list(subgoal.values())[0]
            rel_pos = subgoal_name[0]
            obj1 = subgoal_name[1]
            obj2 = int(subgoal_name[2])
            
            obj1_ids = [node['id'] for node in graph['nodes'] if obj1 in node['class_name']]
            obj_on_table_ids = [dege['from_id'] for dege in graph['edges'] if (dege['relation_type'].lower()==rel_pos.lower()) and (dege['to_id']==obj2)]

            obj1_on_table_ids = list(set(obj1_ids) & set(obj_on_table_ids))
            if obj1_on_table_ids>=subgoal_num:
                subgoal_reward+=1

        if subgoal_reward==len(self.goal):
            return 1
        else:
            return 0

    def watch_tv(self, graph):
        subgoal = self.goal[0]
        assert len(subgoal)==1
        subgoal_name = list(subgoal.keys())[0].split('_')
        tv_id = subgoal_name[0]
        tv_state = subgoal_name[1]
        
        tv_states = [node['states'] for node in graph['nodes'] if tv_id==node['id']]
        assert len(tv_states)==1
        if tv_states==tv_state:
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
    envs = UnityEnv(num_agents=1)
    graph = envs.get_graph()

    ## -------------------------------------------------------------
    ## load task from json, the json file contain max number of objects for each task
    ## -------------------------------------------------------------
    with open('init_pool.json') as file:
        init_pool = json.load(file)

    task_name = random.choice(list(init_pool.keys()))
    goal = {}
    for k,v in init_pool[task_name].items():
        goal[k] = random.randint(0, v)

    ## example setup table
    task_name = 'setup_table'
    goal = {'plates': 2,
            'glasses': 2,
            'wineglass': 1,
            'forks': 0}
    
    ## -------------------------------------------------------------
    ## setup goal based on currect environment
    ## -------------------------------------------------------------
    set_init_goal = SetInitialGoal(goal, obj_position, init_pool[task_name])
    init_graph, env_goal = getattr(set_init_goal, task_name)(graph)
    
    get_reward = GetReward(env_goal)
    reward = getattr(get_reward, task_name)(graph)

    
