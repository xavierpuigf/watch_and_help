import pickle
import pdb
import ipdb
import sys
import os
import random
import json
import numpy as np
import copy
import argparse


curr_dir = os.path.dirname(os.path.abspath(__file__))
home_path = '../../'
sys.path.append(home_path + '/virtualhome')
sys.path.append(f'{curr_dir}/..')

from simulation.unity_simulator import comm_unity
from profilehooks import profile
from init_goal_setter.init_goal_base import SetInitialGoal
from init_goal_setter.tasks import Task


from utils import utils_goals

parser = argparse.ArgumentParser()
parser.add_argument('--num-per-apartment', type=int, default=1, help='Maximum #episodes/apartment')
parser.add_argument('--seed', type=int, default=10, help='Seed for the apartments')

parser.add_argument('--task', type=str, default='setup_table', help='Task name')
parser.add_argument('--apt_str', type=str, default='0,1,2,4,5', help='The apartments where we will generate the data')
parser.add_argument('--port', type=str, default='8092', help='Task name')
parser.add_argument('--display', type=int, default=0, help='Task name')
parser.add_argument('--mode', type=str, default='full', choices=['simple', 'full'], help='Task name')
parser.add_argument('--use-editor', action='store_true', default=False, help='Use unity editor')
parser.add_argument('--exec_file', type=str,
                    default='/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/executables/exec_linux.04.27.x86_64',
                    help='Use unity editor')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed == 0:
        rand = random.Random()
    else:
        rand = random.Random(args.seed)


    with open(f'{curr_dir}/data/init_pool.json') as file:
        init_pool = json.load(file)
    # comm = comm_unity.UnityCommunication()
    if args.use_editor:
        comm = comm_unity.UnityCommunication()
    else:
        comm = comm_unity.UnityCommunication(port=args.port,
                                             file_name=args.exec_file,
                                             no_graphics=True,
                                             logging=False,
                                             x_display=args.display)
    comm.reset()

    # ## show images
    # indices = [-6]
    # _, ncameras = comm.camera_count()
    # cameras_select = list(range(ncameras))
    # cameras_select = [cameras_select[x] for x in indices]
    # (ok_img, imgs) = comm.camera_image(cameras_select, mode='seg_class')
    # import cv2
    # cv2.imwrite('test.png', imgs[0])
    # pdb.set_trace()

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
    with open(f'{curr_dir}/data/class_name_size.json', 'r') as file:
        class_name_size = json.load(file)

    ## -------------------------------------------------------------
    ## gen graph
    ## -------------------------------------------------------------
    task_names = {1: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                  2: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food",
                      "read_book", "watch_tv"],
                  3: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food",
                      "read_book", "watch_tv"],
                  4: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food",
                      "read_book", "watch_tv"],
                  5: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge",
                      "prepare_food"],
                  6: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                  7: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food",
                      "read_book", "watch_tv"]}

    success_init_graph = []

    # task = args.task
    # task = 'setup_table'
    # task = 'put_fridge'
    # task = 'prepare_food'
    # task = 'put_dishwasher'
    # task = 'read_book'
    #apartment_ids = [0,1,2,4,5] # range(7) # TODO: maybe only want the trainign apts
    apartment_ids = [int(apt_id) for apt_id in args.apt_str.split(',')]
    if args.task == 'all':
        tasks = ['setup_table', 'put_fridge', 'prepare_food', 'put_dishwasher', 'read_book']
    else:
        tasks = [args.task]
    num_per_apartment = args.num_per_apartment

    for task in tasks:
        # for apartment in range(6,7):
        for apartment in apartment_ids:
            print('apartment', apartment)

            if task not in task_names[apartment + 1]: continue
            # if apartment != 4: continue
            # apartment = 3

            with open(f'{curr_dir}/data/object_info%s.json' % (apartment + 1), 'r') as file:
                obj_position = json.load(file)

            # pdb.set_trace()bathroomcounter

            # filtering out certain locations
            for obj, pos_list in obj_position.items():
                if obj in ['book', 'remotecontrol']:
                    positions = [pos for pos in pos_list if \
                                 pos[0] == 'INSIDE' and pos[1] in ['kitchencabinets', 'cabinet'] or \
                                 pos[0] == 'ON' and pos[1] in \
                                 (['cabinet', 'bench', 'nightstand'] + ([] if apartment == 2 else ['kitchentable']))]
                elif obj == 'remotecontrol':
                    # TODO: we never get here
                    positions = [pos for pos in pos_list if pos[0] == 'ON' and pos[1] in \
                                 ['tvstand']]
                else:
                    positions = [pos for pos in pos_list if \
                                 pos[0] == 'INSIDE' and pos[1] in ['fridge', 'kitchencabinets', 'cabinet', 'microwave',
                                                                   'dishwasher', 'stove'] or \
                                 pos[0] == 'ON' and pos[1] in \
                                 (['cabinet', 'coffeetable', 'bench'] + ([] if apartment == 2 else ['kitchentable']))]
                obj_position[obj] = positions
            print(obj_position['cutleryfork'])

            num_test = 100000
            count_success = 0
            for i in range(num_test):
                comm.reset(apartment)
                s, original_graph = comm.environment_graph()
                graph = copy.deepcopy(original_graph)

                # pdb.set_trace()
                ## -------------------------------------------------------------
                ## debug
                ## -------------------------------------------------------------
                # debug_function(comm)

                ## -------------------------------------------------------------
                ## choose tasks
                ## -------------------------------------------------------------
                # while True:
                #     task_name = random.choice(task_names[apartment+1])
                #     if task_name in ['read_book', 'watch_tv']:
                #         continue
                #     else:
                #         break
                task_name = task

                print('------------------------------------------------------------------------------')
                print('testing %d/%d: %s. apartment %d' % (i, num_test, task_name, apartment))
                print('------------------------------------------------------------------------------')

                ## -------------------------------------------------------------
                ## setup goal based on currect environment
                ## -------------------------------------------------------------
                set_init_goal = SetInitialGoal(obj_position, class_name_size, init_pool, task_name, same_room=False, rand=rand)
                init_graph, env_goal, success_setup = getattr(Task, task_name)(set_init_goal, graph)
                if env_goal is None:
                    pdb.set_trace()
                if success_setup:
                    # If all objects were well added
                    success, message = comm.expand_scene(init_graph, transfer_transform=False)
                    print('----------------------------------------------------------------------')
                    print(task_name, success, message, set_init_goal.num_other_obj)
                    # print(env_goal)

                    if not success:
                        goal_objs = []
                        goal_names = []
                        for k, goals in env_goal.items():
                            goal_objs += [int(list(goal.keys())[0].split('_')[-1]) for goal in goals if
                                          list(goal.keys())[0].split('_')[-1] not in ['book', 'remotecontrol']]
                            goal_names += [list(goal.keys())[0].split('_')[1] for goal in goals]
                        print(message)
                        obj_names = [obj.split('.')[0] for obj in message['unplaced']]
                        obj_ids = [int(obj.split('.')[1]) for obj in message['unplaced']]
                        id2node = {node['id']: node for node in init_graph['nodes']}

                        for obj_id in obj_ids:
                            print("Objects unplaced")
                            print([id2node[edge['to_id']]['class_name'] for edge in init_graph['edges'] if
                                   edge['from_id'] == obj_id])
                            ipdb.set_trace()
                        if task_name != 'read_book' and task_name != 'watch_tv':
                            intersection = set(obj_names) & set(goal_names)
                        else:
                            intersection = set(obj_ids) & set(goal_objs)

                        ## goal objects cannot be placed
                        if len(intersection) != 0:
                            success2 = False
                        else:
                            init_graph = set_init_goal.remove_obj(init_graph, obj_ids)
                            comm.reset(apartment)
                            success2, message2 = comm.expand_scene(init_graph, transfer_transform=False)
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
                            #comm.reset(apartment)
                            # plate_ids = [node for node in original_graph['nodes'] if node['class_name'] == 'plate']
                            # ith_old_plate = 0
                            # for ith_node, node in enumerate(init_graph['nodes']):
                            #     if node['class_name'] == 'plate':
                            #         if ith_old_plate < len(plate_ids):
                            #             init_graph['nodes'][ith_node] = plate_ids[ith_old_plate]
                            #             ith_old_plate += 1
                            init_graph0 = copy.deepcopy(init_graph)
                            comm.reset(apartment)
                            comm.expand_scene(init_graph, transfer_transform=False)
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
                                            ids = [node['id'] for node in init_graph['nodes'] if
                                                   node['class_name'] == obj_class_name]
                                            print(obj_class_name, v, ids)
                                            # if len(ids) < v:
                                            #     print(obj_class_name, v, ids)
                                            #     pdb.set_trace()

                                count_success += s
                                # if s:
                                # print('-------------------------------------------------------------------------------')
                                # print('-------------------------------------------------------------------------------')
                                # print('-------------------------------------------------------------------------------')
                                # floorid = [tem['id'] for tem in init_graph['nodes'] if 'floor' in tem['class_name']]
                                # print(floorid)
                                # tem2 = [tem['to_id'] for tem in init_graph['edges'] if tem['from_id'] in floorid]
                                # tem3 = [tem['from_id'] for tem in init_graph['edges'] if tem['to_id'] in floorid]
                                # objects = [tem['class_name'] for tem in init_graph['nodes'] if tem['id'] in tem2+tem3]
                                # objectonfloor = set(list(obj_position.keys())).intersection(set(objects))
                                # print(objectonfloor)
                                # print('-------------------------------------------------------------------------------')

                                # print('-------------------------------------------------------------------------------')
                                # print('-------------------------------------------------------------------------------')
                                # print('-------------------------------------------------------------------------------')

                                # objects = []
                                # for node in init_graph['nodes']:
                                #     for edge in init_graph['edges']:
                                #         if (node['id']==edge['from_id']) or (node['id']==edge['to_id']):
                                #             if node not in objects:
                                #                 objects.append(node)

                                # nodes = []
                                # for node in init_graph['nodes']:
                                #     if node not in objects:
                                #         nodes.append(node)

                                # try:
                                #     assert len(nodes)==0
                                # except:
                                #     print(nodes)
                                #     pdb.set_trace()

                                check_result = set_init_goal.check_graph(init_graph, apartment + 1, original_graph)
                                assert check_result == True

                                ipdb.set_trace()
                                success_init_graph.append({'id': count_success,
                                                           'apartment': (apartment + 1),
                                                           'task_name': task_name,
                                                           'init_graph': init_graph,
                                                           'original_graph': original_graph,
                                                           'goal': env_goal})
                else:
                    pdb.set_trace()
                print('apartment: %d: success %d over %d (total: %d)' % (apartment, count_success, i + 1, num_test))
                if count_success >= num_per_apartment:
                    break

    
    data = success_init_graph
    env_task_set = []

    for task in ['setup_table', 'put_fridge', 'put_dishwasher', 'prepare_food', 'read_book']:
        
        for task_id, problem_setup in enumerate(data):
            env_id = problem_setup['apartment'] - 1
            task_name = problem_setup['task_name']
            init_graph = problem_setup['init_graph']
            goal = problem_setup['goal'][task_name]

            goals = utils_goals.convert_goal_spec(task_name, goal, init_graph,
                                                  exclude=['cutleryknife'])
            print('env_id:', env_id)
            print('task_name:', task_name)
            print('goals:', goals)

            task_goal = {}
            for i in range(2):
                task_goal[i] = goals

            env_task_set.append({'task_id': task_id, 'task_name': task_name, 'env_id': env_id, 'init_graph': init_graph,
                                 'task_goal': task_goal,
                                 'level': 0, 'init_rooms': rand.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)})

    pickle.dump(env_task_set, open('dataset/env_task_set_{}_{}.pik'.format(args.num_per_apartment, args.mode), 'wb'))




