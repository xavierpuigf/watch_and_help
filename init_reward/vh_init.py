import pickle
import pdb
import sys
import os
import random
import json
import numpy as np


random.seed(10)

home_path = '../../'
sys.path.append(home_path+'/vh_mdp')
sys.path.append(home_path+'/virtualhome')

from simulation.unity_simulator import comm_unity as comm_unity
from simulation.evolving_graph.utils import load_graph_dict
from profilehooks import profile


class SetInitialGoal:
    def __init__(self, obj_position, class_name_size, init_pool_tasks, task_name):
        self.task_name = task_name
        self.init_pool_tasks = init_pool_tasks
        self.obj_position = obj_position
        self.class_name_size = class_name_size
        self.object_id_count = 1000
        self.surface_size = {}
        self.surface_used_size = {}
        self.max_num_place = 10

        self.min_num_other_object = 15
        self.max_num_other_object = 45
        
        self.set_goal()

    def set_goal(self):
        
        if self.task_name in ['setup_table', 'clean_table', 'put_dishwasher', 'unload_dishwasher', 'put_fridge', 'read_book', 'prepare_food', 'watch_tv']:
            self.init_pool = self.init_pool_tasks[self.task_name]
        
        elif self.task_name=='setup_table_prepare_food':
            self.init_pool = self.init_pool_tasks["setup_table"]
            self.init_pool.update(self.init_pool_tasks["prepare_food"])

        elif self.task_name=='setup_table_read_book':
            self.init_pool = self.init_pool_tasks["setup_table"]
            self.init_pool.update(self.init_pool_tasks["read_book"])
        
        elif self.task_name=='setup_table_watch_tv':
            self.init_pool = self.init_pool_tasks["setup_table"]
            self.init_pool.update(self.init_pool_tasks["watch_tv"])

        elif self.task_name=='setup_table_put_fridge':
            self.init_pool = self.init_pool_tasks["setup_table"]
            self.init_pool.update(self.init_pool_tasks["put_fridge"])

        elif self.task_name=='setup_table_put_dishwasher':
            self.init_pool = self.init_pool_tasks["setup_table"]
            self.init_pool.update(self.init_pool_tasks["put_dishwasher"])

        
        ## make sure the goal is not empty
        while 1:
            self.goal = {}
            for k,v in self.init_pool.items():
                self.goal[k] = random.randint(v['min_num'], v['max_num'])
            
            count = 0
            for k,v in self.goal.items(): 
                count+=v

            if (self.task_name=='read_book' or self.task_name=='watch_tv') and (count!=0):
                break

            if 2<=count<=6:
                break
            
        ## example setup table
        # task_name = 'setup_table'
        # goal = {'plate': 2,
        #         'glasses': 2,
        #         'wineglass': 1,
        #         'cutleryfork': 0}


    def check_goal_achievable(self, graph, comm, env_goal):
        if (self.task_name == 'setup_table') or (self.task_name == 'put_dishwasher') or (self.task_name == 'put_fridge') or (self.task_name == 'prepare_food'):
            for goal in env_goal[self.task_name]:
                # print(self.object_id_count)
                subgoal_name = list(goal.keys())[0]
                num_obj = list(goal.values())[0]
                obj = subgoal_name.split('_')[1]
                target_id = int(subgoal_name.split('_')[3])

                obj_ids = [node['id'] for node in graph['nodes'] if obj == node['class_name']]
                self.remove_obj(graph, obj_ids)
                self.object_id_count = self.add_obj(graph, obj, num_obj, self.object_id_count, self.obj_position, only_position=target_id)
    
            success, message = comm.expand_scene(graph)
        else:
            success = 1
            message = self.task_name

        print(success, message)
        return success


    def convert_size(self, envsize):
        size = envsize[0]*envsize[2]
        return size


    def check_placeable(self, graph, surface_id, obj_name):
        obj_size = self.convert_size(self.class_name_size[obj_name])

        surface_node = [node for node in graph['nodes'] if node['id']==surface_id]
        if surface_id not in self.surface_size:
            surface_node = [node for node in graph['nodes'] if node['id']==surface_id]
            assert len(surface_node)
            self.surface_size[surface_id] = self.convert_size(self.class_name_size[surface_node[0]['class_name']])
        

        if surface_id not in self.surface_used_size:
            objs_on_surface = [edge['from_id'] for edge in graph['edges'] if edge['to_id']==surface_id]
            objs_on_surface_node = [node for node in graph['nodes'] if node['id'] in objs_on_surface]
            objs_on_surface_size = [self.convert_size(self.class_name_size[node['class_name']]) for node in objs_on_surface_node]
            self.surface_used_size[surface_id] = np.sum(objs_on_surface_size) # get size from the initial graph
            

        # print(self.surface_size[surface_id])
        # print(self.surface_used_size[surface_id], obj_size, self.surface_used_size[surface_id]+obj_size)
        # print(obj_name, surface_node[0]['class_name'])


        if self.surface_size[surface_id]/5 > self.surface_used_size[surface_id]+obj_size:
            self.surface_used_size[surface_id] += obj_size
            # print('1')
            return 1
        else:
            # print('0')
            return 0


    def remove_obj(self, graph, obj_ids):
        graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in obj_ids]
        graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in obj_ids and edge['to_id'] not in obj_ids]
        


    def add_obj(self, graph, obj_name, num_obj, object_id, obj_position_pool, only_position=None, except_position=None):
        
        if isinstance(except_position, int):
            except_position = [except_position]
        if isinstance(only_position, int):
            only_position = [only_position]

        edges = []
        nodes = []
        ids_class = {}
        for node in graph['nodes']:
            class_name = node['class_name']
            if class_name not in ids_class: 
                ids_class[class_name] = []
            ids_class[class_name].append(node['id'])
                                
    
        # candidates = [(obj_rel_name[0], obj_rel_name[1]) for obj_rel_name in obj_position_pool[obj_name] if obj_rel_name[1] in ids_class.keys() and (except_position is None or obj_rel_name[1] not in except_position) and (only_position is None or obj_rel_name[1] in only_position)]

        candidates = [(obj_rel_name[0], obj_rel_name[1]) for obj_rel_name in obj_position_pool[obj_name] if obj_rel_name[1] in ids_class.keys()]

        for i in range(num_obj):
            # TODO: we need to check the properties and states, probably the easiest is to get them from the original set of graphs

            num_place = 0
            while 1:
                if num_place > self.max_num_place:
                    break

                if only_position!=None:
                    target_id = random.choice(only_position)
                    
                    target_pool = [k for k,v in ids_class.items() if target_id in v]
                    target_position_pool = [tem[0] for tem in obj_position_pool[obj_name] if tem[1] in target_pool]
                    
                    if len(target_pool)==0 or len(target_position_pool)==0:
                        num_place += 1
                        continue
                    else:
                        relation = random.choice(target_position_pool)
                else:
                    relation, target_classname = random.choice(candidates)
                    target_id = random.choice(ids_class[target_classname])

                    ## target in except_position
                    if (except_position!=None) and (target_id in except_position):
                        num_place += 1
                        continue
                


                ## check if it is possible to put object in this surface
                placeable = self.check_placeable(graph, target_id, obj_name)

                
                # print('placing %s: %dth (total %d), success: %d' % (obj_name, i+1, num_obj, placeable))
                


                if placeable:
                    new_node = {'id': object_id, 'class_name': obj_name, 'properties': ['GRABBABLE'], 'states': [], 'category': 'added_object'}
                    nodes.append(new_node)
                    edges.append({'from_id': object_id, 'relation_type': relation, 'to_id': target_id})
                    object_id += 1
                    break
                else:
                    num_place += 1

                
    


        graph['nodes'] += nodes
        graph['edges'] += edges
        return object_id




    def setup_other_objs(self, graph, object_id, except_position=None):
        new_object_pool = [tem for tem in self.obj_position.keys() if tem not in list(self.goal.keys())] # remove objects in goal

        self.num_other_obj = random.choice(list(range(self.min_num_other_object, self.max_num_other_object+1)))
        for i in range(self.num_other_obj):    
            obj_name = random.choice(new_object_pool)
            obj_in_graph = [node for node in graph['nodes'] if node['class_name']==obj_name] # if the object already in env, skip
            object_id = self.add_obj(graph, obj_name, 1, object_id, self.obj_position, only_position=None, except_position=except_position)

        return object_id



    def set_tv_off(self, graph, tv_id):
        node = [n for n in graph['nodes'] if n['id'] == tv_id]
        assert len(node)==1
        node[0]['states'] = ['OFF']
         # + [state for state in node[0]['states'] if state not in ['ON', 'OFF']]







    def setup_table(self, graph, start=True):
        ## setup table
        # max_num_table = 4
        # num_table = random.randint(1, max_num_table)

        # table_ids = [node['id'] for node in graph['nodes'] if 'table' in node['class_name']]
        # self.remove_obj(graph, table_ids)
        # table_position_pool = self.obj_position['table']
        # self.add_obj(graph, 'table', num_table, table_position_pool)

        table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name']) or ('kitchentable' in node['class_name'])]
        table_id = random.choice(table_ids)

        ## remove objects on table
        objs_on_table = [edge['from_id'] for edge in graph['edges'] if edge['to_id']==table_id]
        self.remove_obj(graph, objs_on_table)

        
        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            self.remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]['env_max_num']+1) # random select objects >= goal
            self.object_id_count = self.add_obj(graph, k, num_obj, self.object_id_count, self.obj_position, except_position=table_id)

        if start:
            self.object_id_count = self.setup_other_objs(graph, self.object_id_count, except_position=table_id)


        ## get goal
        env_goal = {'setup_table': []}
        for k,v in self.goal.items():
            env_goal['setup_table'].append( {'put_{}_on_{}'.format(k, table_id): v} )
        return graph, env_goal






    def clean_table(self, graph, start=True):
        ## clean table
        # max_num_table = 4
        # num_table = random.randint(1, max_num_table)

        # table_ids = [node['id'] for node in graph['nodes'] if 'table' in node['class_name']]
        # self.remove_obj(graph, table_ids)
        # table_position_pool = self.obj_position['table']
        # self.add_obj(graph, 'table', num_table, table_position_pool)
        

        table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name']) or ('kitchentable' in node['class_name'])]
        table_id = random.choice(table_ids)

        ## remove objects on table
        objs_on_table = [edge['from_id'] for edge in graph['edges'] if edge['to_id']==table_id]
        self.remove_obj(graph, objs_on_table)

        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            self.remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]['env_max_num']+1) # random select objects >= goal
            self.object_id_count = self.add_obj(graph, k, v, self.object_id_count, self.obj_position, only_position=table_id) ## add the first v objects on this table
            self.object_id_count = self.add_obj(graph, k, num_obj-v, self.object_id_count, self.obj_position, except_position=table_id) ## add the rest objects on other places
        
        if start:
            self.object_id_count = self.setup_other_objs(graph, self.object_id_count, except_position=table_id)


        ## get goal
        env_goal = {'clean_table': []}
        for k,v in self.goal.items():
            env_goal['clean_table'].append( {'take_{}_off_{}'.format(k, table_id): v} )
        return graph, env_goal


    def put_dishwasher(self, graph, start=True):
        ## setup dishwasher
        # max_num_dishwasher = 4
        # num_dishwasher = random.randint(1, max_num_dishwasher)

        # dishwasher_ids = [node['id'] for node in graph['nodes'] if 'dishwasher' in node['class_name']]
        # self.remove_obj(graph, dishwasher_ids)
        # dishwasher_position_pool = self.obj_position['dishwasher']
        # self.add_obj(graph, 'dishwasher', num_dishwasher, dishwasher_position_pool)
        

        dishwasher_ids = [node['id'] for node in graph['nodes'] if 'dishwasher' in node['class_name']]
        dishwasher_id = random.choice(dishwasher_ids)

        ## remove objects in dishwasher
        objs_in_dishwasher = [edge['from_id'] for edge in graph['edges'] if edge['to_id']==dishwasher_id]
        self.remove_obj(graph, objs_in_dishwasher)

        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            self.remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]['env_max_num']+1) # random select objects >= goal
            self.object_id_count = self.add_obj(graph, k, num_obj, self.object_id_count, self.obj_position, except_position=dishwasher_id)
        
        if start:
            self.object_id_count = self.setup_other_objs(graph, self.object_id_count, except_position=dishwasher_id)


        ## get goal
        env_goal = {'put_dishwasher': []}
        for k,v in self.goal.items():
            env_goal['put_dishwasher'].append( {'put_{}_inside_{}'.format(k, dishwasher_id): v} )
        return graph, env_goal






    def unload_dishwasher(self, graph, start=True):
        ## setup dishwasher
        # max_num_dishwasher = 4
        # num_dishwasher = random.randint(1, max_num_dishwasher)

        # dishwasher_ids = [node['id'] for node in graph['nodes'] if 'dishwasher' in node['class_name']]
        # self.remove_obj(graph, dishwasher_ids)
        # dishwasher_position_pool = self.obj_position['dishwasher']
        # self.add_obj(graph, 'dishwasher', num_dishwasher, dishwasher_position_pool)
        

        dishwasher_ids = [node['id'] for node in graph['nodes'] if 'dishwasher' in node['class_name']]
        dishwasher_id = random.choice(dishwasher_ids)

        ## remove objects in dishwasher
        objs_in_dishwasher = [edge['from_id'] for edge in graph['edges'] if edge['to_id']==dishwasher_id]
        self.remove_obj(graph, objs_in_dishwasher)


        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            self.remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]['env_max_num']+1) # random select objects >= goal
            self.object_id_count = self.add_obj(graph, k, v, self.object_id_count, self.obj_position, only_position=dishwasher_id) ## add the first v objects on this table
            self.object_id_count = self.add_obj(graph, k, num_obj-v, self.object_id_count, self.obj_position, except_position=dishwasher_id) ## add the rest objects on other places
        
        if start:
            self.object_id_count = self.setup_other_objs(graph, self.object_id_count, except_position=dishwasher_id)


        ## get goal
        env_goal = {'unload_dishwasher': []}
        for k,v in self.goal.items():
            env_goal['unload_dishwasher'].append( {'take_{}_from_{}'.format(k, dishwasher_id): v} )
        return graph, env_goal



    def put_fridge(self, graph, start=True):
        ## setup fridge
        # max_num_fridge = 4
        # num_fridge = random.randint(1, max_num_fridge)

        # fridge_ids = [node['id'] for node in graph['nodes'] if 'fridge' in node['class_name']]
        # self.remove_obj(graph, fridge_ids)
        # fridge_position_pool = self.obj_position['fridge']
        # self.add_obj(graph, 'fridge', num_fridge, fridge_position_pool)
        

        fridge_ids = [node['id'] for node in graph['nodes'] if 'fridge' in node['class_name']]
        fridge_id = random.choice(fridge_ids)

        ## remove objects in fridge
        objs_in_fridge = [edge['from_id'] for edge in graph['edges'] if edge['to_id']==fridge_id]
        self.remove_obj(graph, objs_in_fridge)

        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            self.remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]['env_max_num']+1) # random select objects >= goal
            self.object_id_count = self.add_obj(graph, k, num_obj, self.object_id_count, self.obj_position, except_position=fridge_id)
        
        if start:
            self.object_id_count = self.setup_other_objs(graph, self.object_id_count, except_position=fridge_id)


        ## get goal
        env_goal = {'put_fridge': []}
        for k,v in self.goal.items():
            env_goal['put_fridge'].append( {'put_{}_inside_{}'.format(k, fridge_id): v} )
        return graph, env_goal



    def read_book(self, graph, start=True):
        max_num_book = self.init_pool['book']['env_max_num']
        num_book = random.randint(self.goal['book'], max_num_book+1)

        book_ids = [node['id'] for node in graph['nodes'] if 'book' in node['class_name']]
        self.remove_obj(graph, book_ids)
        self.object_id_count = self.add_obj(graph, 'book', num_book, self.object_id_count, self.obj_position)
        

        book_ids = [node['id'] for node in graph['nodes'] if 'book' in node['class_name']]
        book_id = random.choice(book_ids)

        if start:
            self.object_id_count = self.setup_other_objs(graph, self.object_id_count)

        ## get goal
        env_goal = {'read_book': [{'read_{}'.format(book_id)}]}
        return graph, env_goal


    def prepare_food(self, graph, start=True):
        # max_num_table = 4
        # num_table = random.randint(1, max_num_table)

        # table_ids = [node['id'] for node in graph['nodes'] if 'table' in node['class_name']]
        # self.remove_obj(graph, table_ids)
        # table_position_pool = self.obj_position['table']
        # self.add_obj(graph, 'table', num_table, table_position_pool)
        

        table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name']) or ('kitchentable' in node['class_name'])]
        table_id = random.choice(table_ids)

        ## remove objects on table
        objs_on_table = [edge['from_id'] for edge in graph['edges'] if edge['to_id']==table_id]
        self.remove_obj(graph, objs_on_table)


        for k,v in self.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            self.remove_obj(graph, obj_ids)

            num_obj = random.randint(v, self.init_pool[k]['env_max_num']+1) # random select objects >= goal
            self.object_id_count = self.add_obj(graph, k, num_obj, self.object_id_count, self.obj_position, except_position=table_id)
        
        if start:
            self.object_id_count = self.setup_other_objs(graph, self.object_id_count, except_position=table_id)


        ## get goal
        env_goal = {'prepare_food': []}
        for k,v in self.goal.items():
            env_goal['prepare_food'].append( {'put_{}_on_{}'.format(k, table_id): v} )
        return graph, env_goal


    def watch_tv(self, graph, start=True):
        # max_num_tv = 4
        # num_tv = random.randint(1, max_num_tv)

        # tv_ids = [node['id'] for node in graph['nodes'] if 'tv' in node['class_name']]
        # self.remove_obj(graph, tv_ids)
        # tv_position_pool = self.obj_position['tv']
        # self.add_obj(graph, 'tv', num_tv, tv_position_pool)
        
        
        tv_ids = [node['id'] for node in graph['nodes'] if 'tv' in node['class_name']]
        tv_id = random.choice(tv_ids)

        self.set_tv_off(graph, tv_id)

        if start:
            self.object_id_count = self.setup_other_objs(graph, self.object_id_count)

        ## get goal
        env_goal = {'watch_tv': [{'{}_on'.format(tv_id)}]}
        return graph, env_goal


    def setup_table_prepare_food(self, graph):
        graph, env_goal1 = self.setup_table(graph)
        graph, env_goal2 = self.prepare_food(graph, start=False)
        return graph, env_goal1.update(env_goal2)

    def setup_table_read_book(self, graph):
        graph, env_goal1 = self.setup_table(graph)
        graph, env_goal2 = self.read_book(graph, start=False)
        return graph, env_goal1.update(env_goal2)
    
    def setup_table_watch_tv(self, graph):
        graph, env_goal1 = self.setup_table(graph)
        graph, env_goal2 = self.watch_tv(graph, start=False)
        return graph, env_goal1.update(env_goal2)

    def setup_table_put_fridge(self, graph):
        graph, env_goal1 = self.setup_table(graph)
        graph, env_goal2 = self.put_fridge(graph, start=False)
        return graph, env_goal1.update(env_goal2)

    def setup_table_put_dishwasher(self, graph):
        graph, env_goal1 = self.setup_table(graph)
        graph, env_goal2 = self.put_dishwasher(graph, start=False)
        return graph, env_goal1.update(env_goal2)


def debug_function(comm):
    with open('data/object_info.json', 'r') as file:
        obj_position = json.load(file)


    success_edges = []
    fail_target_nodes = []

    for obj_name in obj_position['objects_grab']:
        object_id = 2000
        new_node = {'id': object_id, 'class_name': obj_name, 'properties': ['GRABBABLE'], 'states': [], 'category': 'added_object'}
        nodes = [new_node]

        for target_name in obj_position['objects_inside']:
            comm.reset()
            s, graph = comm.environment_graph()


            target_node = [node for node in graph['nodes'] if node['class_name']==target_name]
            if len(target_node)==0:
                print(target_name)
                fail_target_nodes.append(target_name)
                continue

            target_id = target_node[0]['id']

            edges = [{'from_id': object_id, 'relation_type': 'INSIDE', 'to_id': target_id}]

            
            graph['nodes'] += nodes
            graph['edges'] += edges
            success, message = comm.expand_scene(graph)
            # print(success, message)

            if success:
                success_edges.append({'from_id': obj_name, 'relation_type': 'INSIDE', 'to_id': target_name})
            else:
                print({'from_id': obj_name, 'relation_type': 'INSIDE', 'to_id': target_name})

        for target_name in obj_position['objects_surface']:
            comm.reset()
            s, graph = comm.environment_graph()


            target_node = [node for node in graph['nodes'] if node['class_name']==target_name]
            if len(target_node)==0:
                print(target_name)
                fail_target_nodes.append(target_name)
                continue

            target_id = target_node[0]['id']

            edges = [{'from_id': object_id, 'relation_type': 'ON', 'to_id': target_id}]

            
            graph['nodes'] += nodes
            graph['edges'] += edges
            success, message = comm.expand_scene(graph)
            # print(success, message)

            if success:
                success_edges.append({'from_id': obj_name, 'relation_type': 'ON', 'to_id': target_name})
            else:
                print({'from_id': obj_name, 'relation_type': 'ON', 'to_id': target_name})

    
    # with open('data/object_info_7.json', 'w') as file:
    #     json.dump(success_edges, file)


    # ## load file and save
    # with open('data/object_info_%s.json'%apartment, 'r') as file:
    #     obj_position = json.load(file)

    # objs = {}
    # for rel in obj_position:
    #     from_obj = rel['from_id']
    #     relation_type = rel['relation_type']
    #     to_obj = rel['to_id']

    #     if from_obj not in objs:
    #         objs[from_obj] = []
    #     objs[from_obj].append([relation_type, to_obj])

    # with open('data/object_info%s.json'%apartment, 'w') as file:
    #     json.dump(objs, file)

    


if __name__ == "__main__":
    # Better to not sue UnityEnv here, it is faster and it allows to create an env without agents

    ## -------------------------------------------------------------
    ## load task from json, the json file contain max number of objects for each task
    ## -------------------------------------------------------------
    with open('data/init_pool.json') as file:
        init_pool = json.load(file)



    comm = comm_unity.UnityCommunication()
    comm.reset()
    s, graph = comm.environment_graph()
    

    ## -------------------------------------------------------------
    ## get object sizes
    ## -------------------------------------------------------------

    ## step1 write object size of each apartment
    # class_name_size = {node['class_name']: node['bounding_box']['size'] for node in graph['nodes']}
    # with open('class_name_size7.json', 'w') as file:
    #     json.dump(class_name_size, file)
    
    ## -------------------------------------------------------------
    ## step2 combine object size from each apartment
    # class_name_size = {}
    # for i in range(7):
    #     with open('data/class_name_size%s.json' % str(i+1), 'r') as file:
    #         class_name_size.update(json.load(file))

    # class_name = np.unique(list(class_name_size.keys()))
    # class_name_size = {tem: class_name_size[tem] for tem in class_name}

    # with open('data/class_name_size.json', 'w') as file:
    #     json.dump(class_name_size, file)

    ## -------------------------------------------------------------
    ## step3 load object size
    with open('data/class_name_size.json', 'r') as file:
        class_name_size = json.load(file)

    ## -------------------------------------------------------------
    ## gen graph
    ## -------------------------------------------------------------
    task_names = {  1: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    2: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    3: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    4: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    5: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    6: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    7: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"]}
                    

    success_init_graph = []

    for apartment in range(7):
        # apartment = 3

        with open('data/object_info%s.json'%(apartment+1), 'r') as file:
            obj_position = json.load(file)


        num_test = 100000
        count_success = 0
        for i in range(num_test):
            comm.reset(apartment)
            s, graph = comm.environment_graph()


            ## -------------------------------------------------------------
            ## debug
            ## -------------------------------------------------------------
            # debug_function(comm)
            


            ## -------------------------------------------------------------
            ## choose tasks
            ## -------------------------------------------------------------
            task_name = random.choice(task_names[apartment+1])
            # task_name = 'setup_table'


            print('------------------------------------------------------------------------------')
            print('testing %d: %s' % (i, task_name))
            print('------------------------------------------------------------------------------')
            
            ## -------------------------------------------------------------
            ## setup goal based on currect environment
            ## -------------------------------------------------------------
            set_init_goal = SetInitialGoal(obj_position, class_name_size, init_pool, task_name)
            init_graph, env_goal = getattr(set_init_goal, task_name)(graph)
            
            success, message = comm.expand_scene(init_graph)
            print(task_name, success, message, set_init_goal.num_other_obj)
            # print(env_goal)


            if success:
                success = set_init_goal.check_goal_achievable(init_graph, comm, env_goal)
                count_success += success

                if success:
                    success_init_graph.append({'apartment': (apartment+1),
                                                'task_name': task_name,
                                                'init_graph': init_graph,
                                                'goal': env_goal})


            print('apartment: %d: success %d over %d (total: %d)' % (apartment, count_success, i+1, num_test) )

            if count_success>=50:
                break
    
    pdb.set_trace()
    pickle.dump( success_init_graph, open( "result/init7_50.p", "wb" ) )
    # tem = pickle.load( open( "result/init100.p", "rb" ) )




        