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


def remove_obj(graph, obj_ids):
    graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in obj_ids]
    graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in obj_ids and edge['to_id'] not in obj_ids]
    
def add_obj(graph, obj_name, num_obj, obj_position_pool, only_position=None, except_position=None):
    pass

def setup_other_objs(graph):
    pass

def set_tv_off(graph, tv_id):
    node = utils_unity_graph.find_nodes(graph, id=tv_id)
    node['states'] = 'OFF' + [state for state in node['states'] if node['states'] not in ['ON', 'OFF']]

class SetInitialGoal:
    def __init__(self, goal, obj_position, init_pool):
        self.goal = goal
        self.obj_position
        self.init_pool = init_pool

    def setup_table(self, graph):
        ## setup table
        max_num_table = 4
        num_table = random.randint(1, max_num_table)

        table_ids = [node['id'] for node in graph['nodes'] if 'tables' in node['class_name']]
        remove_obj(graph, table_ids)
        table_position_pool = self.obj_position['table']
        add_obj(graph, 'table', num_table, table_position_pool)
        

        table_ids = [node['id'] for node in graph['nodes'] if 'tables' in node['class_name']]
        table_id = random.choice(table_ids)

        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]) # random select objects >= goal
            obj_position_pool = self.obj_position[k]
            add_obj(graph, k, num_obj, obj_position_pool, except_position=table_id)
        
        setup_other_objs(graph)


        ## get goal
        env_goal = []
        for k,v in self.goal.items():
            env_goal.append( {'on_{}_{}'.format(k, table_id): v} )
        return graph, env_goal






    def clean_table(self, graph):
        ## clean table
        max_num_table = 4
        num_table = random.randint(1, max_num_table)

        table_ids = [node['id'] for node in graph['nodes'] if 'tables' in node['class_name']]
        remove_obj(graph, table_ids)
        table_position_pool = self.obj_position['table']
        add_obj(graph, 'table', num_table, table_position_pool)
        

        table_ids = [node['id'] for node in graph['nodes'] if 'tables' in node['class_name']]
        table_id = random.choice(table_ids)

        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]) # random select objects >= goal
            obj_position_pool = self.obj_position[k]
            add_obj(graph, k, v, obj_position_pool, only_position=table_id) ## add the first v objects on this table
            add_obj(graph, k, num_obj-v, obj_position_pool, except_position=table_id) ## add the rest objects on other places
        
        setup_other_objs(graph)


        ## get goal
        env_goal = []
        for k,v in self.goal.items():
            env_goal.append( {'off_{}_{}'.format(k, table_id): v} )
        return graph, env_goal


    def put_diswasher(self, graph):
        ## setup diswasher
        max_num_diswasher = 4
        num_diswasher = random.randint(1, max_num_diswasher)

        diswasher_ids = [node['id'] for node in graph['nodes'] if 'diswasher' in node['class_name']]
        remove_obj(graph, diswasher_ids)
        diswasher_position_pool = self.obj_position['diswasher']
        add_obj(graph, 'diswasher', num_diswasher, diswasher_position_pool)
        

        diswasher_ids = [node['id'] for node in graph['nodes'] if 'diswasher' in node['class_name']]
        diswasher_id = random.choice(diswasher_ids)

        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]) # random select objects >= goal
            obj_position_pool = self.obj_position[k]
            add_obj(graph, k, num_obj, obj_position_pool, except_position=diswasher_id)
        
        setup_other_objs(graph)


        ## get goal
        env_goal = []
        for k,v in self.goal.items():
            env_goal.append( {'on_{}_{}'.format(k, diswasher_id): v} )
        return graph, env_goal






    def unload_diswasher(self, graph):
        ## setup diswasher
        max_num_diswasher = 4
        num_diswasher = random.randint(1, max_num_diswasher)

        diswasher_ids = [node['id'] for node in graph['nodes'] if 'diswasher' in node['class_name']]
        remove_obj(graph, diswasher_ids)
        diswasher_position_pool = self.obj_position['diswasher']
        add_obj(graph, 'diswasher', num_diswasher, diswasher_position_pool)
        

        diswasher_ids = [node['id'] for node in graph['nodes'] if 'diswasher' in node['class_name']]
        diswasher_id = random.choice(diswasher_ids)


        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]) # random select objects >= goal
            obj_position_pool = self.obj_position[k]
            add_obj(graph, k, v, obj_position_pool, only_position=diswasher_id) ## add the first v objects on this table
            add_obj(graph, k, num_obj-v, obj_position_pool, except_position=diswasher_id) ## add the rest objects on other places
        
        setup_other_objs(graph)


        ## get goal
        env_goal = []
        for k,v in self.goal.items():
            env_goal.append( {'off_{}_{}'.format(k, diswasher_id): v} )
        return graph, env_goal



    def put_fridge(self, graph):
       ## setup fridge
        max_num_fridge = 4
        num_fridge = random.randint(1, max_num_fridge)

        fridge_ids = [node['id'] for node in graph['nodes'] if 'fridge' in node['class_name']]
        remove_obj(graph, fridge_ids)
        fridge_position_pool = self.obj_position['fridge']
        add_obj(graph, 'fridge', num_fridge, fridge_position_pool)
        

        fridge_ids = [node['id'] for node in graph['nodes'] if 'fridge' in node['class_name']]
        fridge_id = random.choice(fridge_ids)

        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]) # random select objects >= goal
            obj_position_pool = self.obj_position[k]
            add_obj(graph, k, num_obj, obj_position_pool, except_position=fridge_id)
        
        setup_other_objs(graph)


        ## get goal
        env_goal = []
        for k,v in self.goal.items():
            env_goal.append( {'on_{}_{}'.format(k, fridge): v} )
        return graph, env_goal



    def read_book(self, graph):
        max_num_book = self.init_pool['book']
        num_book = random.randint(1, max_num_book)

        book_ids = [node['id'] for node in graph['nodes'] if 'book' in node['class_name']]
        remove_obj(graph, book_ids)
        book_position_pool = self.obj_position['book']
        add_obj(graph, 'book', num_table, book_position_pool)
        

        book_ids = [node['id'] for node in graph['nodes'] if 'book' in node['class_name']]
        book_id = random.choice(book_ids)

        setup_other_objs(graph)

        ## get goal
        env_goal = [{'grab_{}'.format(book_id)}]
        return graph, env_goal


    def prepare_food(self, graph):
        max_num_table = 4
        num_table = random.randint(1, max_num_table)

        table_ids = [node['id'] for node in graph['nodes'] if 'tables' in node['class_name']]
        remove_obj(graph, table_ids)
        table_position_pool = self.obj_position['table']
        add_obj(graph, 'table', num_table, table_position_pool)
        

        table_ids = [node['id'] for node in graph['nodes'] if 'tables' in node['class_name']]
        table_id = random.choice(table_ids)


        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]) # random select objects >= goal
            obj_position_pool = self.obj_position[k]
            add_obj(graph, k, num_obj, obj_position_pool, except_position=table_id)
        
        setup_other_objs(graph)


        ## get goal
        env_goal = []
        for k,v in self.goal.items():
            env_goal.append( {'on_{}_{}'.format(k, table_id): v} )
        return graph, env_goal


    def watch_tv(self, graph):
        max_num_tv = 4
        num_tv = random.randint(1, max_num_tv)

        tv_ids = [node['id'] for node in graph['nodes'] if 'tv' in node['class_name']]
        remove_obj(graph, tv_ids)
        tv_position_pool = self.obj_position['tv']
        add_obj(graph, 'tv', num_tv, tv_position_pool)
        

        tv_ids = [node['id'] for node in graph['nodes'] if 'tv' in node['class_name']]
        tv_id = random.choice(tv_ids)

        set_tv_off(tv_id)
        setup_other_objs()

        ## get goal
        env_goal = [{'{}_on'.format(tv_id)}]
        return graph, env_goal





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
    
    



    