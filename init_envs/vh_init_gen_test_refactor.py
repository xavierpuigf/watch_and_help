import pickle
import pdb
import sys
import os
import random
import json
import numpy as np
import copy
import argparse
import pickle as pkl

random.seed(10)

home_path = '../../'
sys.path.append(home_path+'/virtualhome')

from simulation.unity_simulator import comm_unity as comm_unity
from profilehooks import profile
from init_goal_setter.init_goal_base import SetInitialGoal
from init_goal_setter.tasks import Task

parser = argparse.ArgumentParser()
parser.add_argument('--num-per-task', type=int, default=10, help='Maximum #episodes/task')
parser.add_argument('--num-per-apartment', type=int, default=10, help='Maximum #episodes/apartment')
parser.add_argument('--task', type=str, default='setup_table', help='Task name')
parser.add_argument('--demo-id', type=int, default=0, help='demo index')
parser.add_argument('--port-number', type=int, default=8290, help='port')
parser.add_argument('--display', type=str, default='2', help='display')
parser.add_argument('--exec_file', type=str, default='/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/executables/exec_linux.04.27.x86_64', help='Use unity editor')

if __name__ == "__main__":
    args = parser.parse_args()
    # Better to not sue UnityEnv here, it is faster and it allows to create an env without agents

    ## -------------------------------------------------------------
    ## load task from json, the json file contain max number of objects for each task
    ## -------------------------------------------------------------
    with open('data/init_pool.json') as file:
        init_pool = json.load(file)

    # file_split = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/data_challenge/split/watch_scenes_split.json'
    file_split = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/data_challenge/split/watch_scenes_mutliple_split.json'

    with open(file_split, 'r') as f:
        content_split = json.load(f)

    # Predicates train and test
    cache_files = []
    predicates = {'train': [], 'test': []}

    # for elem in content_split['train']:
    #     if elem['pred_str'] not in cache_files:
    #         cache_files.append(elem['pred_str'])
    #     else:
    #         continue
    #     predicates['train'].append((elem['pred_dict'], elem['task_name'], elem['pred_str']))

    for elem in content_split['test']:
        # Create a has pred str
        pred_dict = elem['pred_dict']
        pred_str = ','.join(sorted([x+'.'+str(y) for x,y in pred_dict.items()]))

        if pred_str not in cache_files:
            cache_files.append(elem['pred_str'])
        else:
            pdb.set_trace()
            continue
        predicates['test'].append((elem['pred_dict'], elem['task_name'], pred_str))

    print("Done")
    pdb.set_trace()
    # args.dataset_path = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/vh_multiagent_models/analysis/info_demo_scenes_posteccv.json'
    args.record_dir = '../initial_environments/data/init_envs'
    # data = json.load(open(args.dataset_path, 'r'))
    # json_file_list = data['split']['test_prog']

    simulator_args={
                   'file_name': args.exec_file,
                    'x_display': 0,
                    'logging': False,
                    'no_graphics': True,
                }
    comm = comm_unity.UnityCommunication(port=str(args.port_number), **simulator_args)
    comm.reset()
    s, graph = comm.environment_graph()
    

    
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
                    5: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food"],
                    6: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    7: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"]}
                    

    success_init_graph = []

    apartment_list = [3, 6] # [0, 1, 2, 4, 5] # [3, 6]

    task_counts = {"setup_table": 0, 
                   "put_dishwasher": 0, 
                   "put_fridge": 0, 
                   "prepare_food": 0,
                   "read_book": 0}

    test_set = []

    task_id = 0



    for predicates_dict, task_name, pred_str in predicates['test']: # test
        # json_path = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/vh_multiagent_models' + \
        #             json_file[2:]
        # if json_path.endswith('json'):
        #     content = json.load(open(json_path, 'r'))
        # else:
        #     with open(json_path, 'rb') as f:
        #         content = pkl.load(f)
        #env_id = content['env_id']
        #task_name = content['task_name']
        demo_goals = predicates_dict
        # {predicate: count for predicate, count in predicates_dict.items() if predicate.startswith('on') or predicate.startswith('inside')}
        #print('env_id:', env_id)
        print('task_name:', task_name)
        print('goals:', demo_goals)
        task_goal = {}
        for i in range(2):
            task_goal[i] = demo_goals

        goal_class = {}
        pdb.set_trace()
        # id2node = {node['id']: node for node in content['init_unity_graph']['nodes']}
        for predicate, count in task_goal[0].items():
            elements = predicate.split('_')
            if elements[2].isdigit():
                pdb.set_trace()
                new_predicate = '{}_{}_{}'.format(elements[0], elements[1], id2node[int(elements[2])]['class_name'])
                location_name = id2node[int(elements[2])]['class_name']
            else:
                new_predicate = predicate
            print(new_predicate)
            goal_class[new_predicate] = count

        # pdb.set_trace()
        # if task_counts[task_name] >= args.num_per_task: continue

        num_test = 10
        count_success = 0

        for i in range(num_test):
            # Select apartments that allow the task
            apt_list = [capt for capt in apartment_list if task_name in task_names[capt+1]]
            assert(len(apt_list) > 0)
            apartment = random.choice(apt_list)
            comm.reset(apartment)
            s, original_graph = comm.environment_graph()
            graph = copy.deepcopy(original_graph)

            with open('data/object_info%s.json'%(apartment+1), 'r') as file:
                obj_position = json.load(file)

            # pdb.set_trace()bathroomcounter

            # filtering out certain locations
            for obj, pos_list in obj_position.items():
                if obj in ['book', 'remotecontrol']:
                    positions = [pos for pos in pos_list if \
                    pos[0] == 'INSIDE' and pos[1] in ['kitchencabinet', 'cabinet'] or \
                    pos[0] == 'ON' and pos[1] in \
                    (['cabinet', 'bench', 'nightstand'] + ([] if apartment == 2 else ['kitchentable']))]
                elif obj == 'remotecontrol':
                     positions = [pos for pos in pos_list if pos[0] == 'ON' and pos[1] in \
                    ['tvstand']]
                else:
                    positions = [pos for pos in pos_list if \
                    pos[0] == 'INSIDE' and pos[1] in ['fridge', 'kitchencabinets', 'cabinet', 'microwave', 'dishwasher', 'stove'] or \
                    pos[0] == 'ON' and pos[1] in \
                        (['cabinet', 'coffeetable', 'bench']+ ([] if apartment == 2 else ['kitchentable']))]
                obj_position[obj] = positions
            print(obj_position['cutleryfork'])


            print('------------------------------------------------------------------------------')
            print('testing %d: %s' % (i, task_name))
            print('------------------------------------------------------------------------------')
            
            ## -------------------------------------------------------------
            ## setup goal based on currect environment
            ## -------------------------------------------------------------
            pdb.set_trace()
            set_init_goal = SetInitialGoal(obj_position, class_name_size, init_pool, task_name, same_room=False, goal_template=demo_goals)
            # pdb.set_trace()
            init_graph, env_goal, success_expand = getattr(Task, task_name)(set_init_goal, graph)

            
            if success_expand:
                
                success, message = comm.expand_scene(init_graph)
                print('----------------------------------------------------------------------')
                print(task_name, success, message, set_init_goal.num_other_obj)
                # print(env_goal)

                if not success:
                    goal_objs = []
                    goal_names = []
                    for k,goals in env_goal.items():
                        goal_objs += [int(list(goal.keys())[0].split('_')[-1]) for goal in goals if list(goal.keys())[0].split('_')[-1] not in ['book', 'remotecontrol']]
                        goal_names += [list(goal.keys())[0].split('_')[1] for goal in goals]
                    
                    obj_names = [obj.split('.')[0] for obj in message['unplaced']]
                    obj_ids = [int(obj.split('.')[1]) for obj in message['unplaced']]
                    id2node = {node['id']: node for node in init_graph['nodes']}
                    for obj_id in obj_ids:
                        print([id2node[edge['to_id']]['class_name'] for edge in init_graph['edges'] if edge['from_id'] == obj_id])

                    if task_name!='read_book' and task_name!='watch_tv':
                        intersection = set(obj_names) & set(goal_names)
                    else:
                        intersection = set(obj_ids) & set(goal_objs)
                    
                    ## goal objects cannot be placed
                    if len(intersection)!=0:
                        success2 = False
                    else:
                        init_graph = set_init_goal.remove_obj(init_graph, obj_ids)
                        success2, message2 = comm.expand_scene(init_graph)
                        success = True
                
                else:
                    success2 = True
                    

                if success2 and success:
                    # if apartment == 4:
                    #     init_graph = set_init_goal.remove_obj(init_graph, [348])
                    # elif apartment == 6:
                    #     init_graph = set_init_goal.remove_obj(init_graph, [173])

                    success = set_init_goal.check_goal_achievable(init_graph, comm, env_goal, apartment)

                    if success:
                        comm.reset(apartment)
                        # plate_ids = [node for node in original_graph['nodes'] if node['class_name'] == 'plate']
                        # ith_old_plate = 0
                        # for ith_node, node in enumerate(init_graph['nodes']):
                        #     if node['class_name'] == 'plate':
                        #         if ith_old_plate < len(plate_ids):
                        #             init_graph['nodes'][ith_node] = plate_ids[ith_old_plate]
                        #             ith_old_plate += 1
                        init_graph0 = copy.deepcopy(init_graph)
                        comm.expand_scene(init_graph)
                        s, init_graph = comm.environment_graph()
                        print('final s:', s)
                        if s:
                            for subgoal in env_goal[task_name]:
                                for k, v in subgoal.items():
                                    elements = k.split('_')
                                    # print(elements)
                                    # pdb.set_trace()
                                    if len(elements) == 4:
                                        obj_class_name = elements[1]
                                        ids = [node['id'] for node in init_graph['nodes'] if node['class_name'] == obj_class_name]
                                        print(obj_class_name, v, ids)
                                        if len(ids) < v:
                                            print(obj_class_name, v, ids)
                                            s = 0
                                            break
                                        #     pdb.set_trace()

                            count_success += s
                        if s:
                            cur_goal_spec = {}

                            for predicate, count in task_goal[0].items():
                                elements = predicate.split('_')
                                # pdb.set_trace()
                                # Add ids
                                if elements[2].isdigit():
                                    location_id = list([node['id'] for node in init_graph['nodes'] if node['class_name'] == location_name])[0]
                                    new_predicate = '{}_{}_{}'.format(elements[0], elements[1], location_id)
                                    cur_goal_spec[new_predicate] = count
                                else:
                                    if elements[2] == 'character':
                                        location_id = 1
                                    else:
                                        location_name = elements[2]
                                        location_id = list([node['id'] for node in init_graph['nodes'] if
                                                            node['class_name'] == location_name])[0]
                                    new_predicate = '{}_{}_{}'.format(elements[0], elements[1], location_id)
                                    cur_goal_spec[new_predicate] = count

                            # pdb.set_trace()
                            test_set.append({'task_id': task_id, 
                                      'task_name': task_name, 
                                      'env_id': apartment, 
                                      'init_graph': init_graph, 
                                      'task_goal': {0: cur_goal_spec, 1: cur_goal_spec},
                                      'goal_class': goal_class,
                                      'level': 1, 
                                      'init_rooms': random.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2),
                                      'pred_str': pred_str})
                            print('task_name:', test_set[-1]['task_name'])
                            print('task_goal:', test_set[-1]['task_goal'])
                            task_id += 1
                            # pdb.set_trace()


            print('apartment: %d: success %d over %d (total: %d)' % (apartment, count_success, i+1, num_test) )

            if count_success>=1:
                task_counts[task_name] += 1
                break
    
    # pdb.set_trace()
    print(len(test_set))
    print(task_counts)
    pickle.dump(test_set, open(args.record_dir + '/test_env_set_help_{}_multitask_neurips.pik'.format(args.num_per_task), 'wb'))



        