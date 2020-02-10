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



class GetReward:
    def __init__(self, goal):
        self.goal = goal

    def setup_table(graph):
        pdb.set_trace()
        for k,v in self.goal.items():
            table_ids = [node['id'] for node in graph['nodes'] if 'tables' in node['class_name']]

        if (state_num_people >= goal['num_people']) & \
            (state_num_plates >= goal['num_plates']) & \
            (state_num_glasses >= goal['num_glasses']) & \
            (state_num_wine >= goal['num_wine']) & \
            (state_num_forks >= goal['num_forks']):
            return 1
        else:
            return 0
        

    def read_book(graph):
        if (state_num_book >= goal['num_book']):
            return 1
        else:
            return 0

    def clean_table(graph):
        if (state_num_food >= goal['num_food']) & \
            (state_num_plate >= goal['num_plate']) & \
            (state_num_glasses >= goal['num_glasses']) & \
            (state_num_wine >= goal['num_wine']) & \
            (state_num_forks >= goal['num_forks']):
            return 1
        else:
            return 0

        
    def put_diswasher(graph):
        if (state_num_plate >= goal['num_plate']) & \
            (state_num_glasses >= goal['num_glasses']) & \
            (state_num_forks >= goal['num_forks']):
            return 1
        else:
            return 0


    def unload_diswasher(graph):
        if (state_num_plate >= goal['num_plate']) & \
            (state_num_glasses >= goal['num_glasses']) & \
            (state_num_forks >= goal['num_forks']):
            return 1
        else:
            return 0

    def put_fridge(graph):
        if (state_num_food >= goal['num_food']):
            return 1
        else:
            return 0

    def prepare_food(graph):
        state_num_food >= goal['num_food']
        FOOD_APPLE
        FOOD_CEREAL
        FOOD_BANANA
        FOOD_BREAD
        FOOD_CARROT
        FOOD_CHICKEN
        FOOD_DESSERT
        FOOD_FISH
        FOOD_HAMBURGER
        FOOD_LEMON
        FOOD_LIME
        FOOD_OATMEAL
        FOOD_POTATO
        FOOD_SALT
        FOOD_SNACK
        FOOD_SUGAR
        FOOD_TURKEY

    def watch_tv(graph):
        goal['tv'] = 'off'





    # goals = [
    #     (setup_table, ),
    #     (read_book, ),
    #     (clean_table, ),
    #     (put_diswasher, ),
    #     (unload_diswasher, ),
    #     (put_fridge, ),
    #     (prepare_food, ),
    #     (watch_tv, ),

    #     (setup_table, prepare_food),
    #     (setup_table, read_book),
    #     (setup_table, watch_tv),
    #     (setup_table, put_fridge),
    #     (setup_table, put_fridge),
    #     (setup_table, put_diswasher),
    # ]

if __name__ == "__main__":
    envs = UnityEnv(num_agents=1)
    graph = envs.get_graph()

    
    with open('obj_position.json') as file:
        obj_position = json.load(file)


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

    
