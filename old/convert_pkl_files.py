import ipdb
import pickle as pkl
import copy
import sys
from tqdm import tqdm
import numpy as np
sys.path.append('../virtualhome/simulation/')
from unity_simulator import comm_unity
import numpy.linalg

def remove_floor(updated_graph, curr_graph):
    # translate_prefab = {
    #     'Cupcake_1': 'DHP_PRE_Pink_cupcake_1024',
    #     'Cupcake_2': 'DHP_PRE_Rainbow_cupcake_1024',
    #     'PRE_PRO_Box_02': 'PRE_PRO_Box_01'
    # }
    # ipdb.set_trace()
    # updated_graph = copy.deepcopy(updated_graph)
    floor_none_ids = sorted([node['id'] for node in updated_graph['nodes'] if node['prefab_name'] == "Floor" or 
                             node['prefab_name'] == 'mH_Bucket01_s' or "MainDoor" in node['prefab_name']] )
    # print(floor_none_ids)
    node_id_mapping = {}
    updated_graph['nodes'] = [node for node in updated_graph['nodes'] if node['id'] not in floor_none_ids]
    updated_graph['edges'] = [edge for edge in updated_graph['edges'] if edge['from_id'] not in floor_none_ids and edge['to_id'] not in floor_none_ids]
    
    for node in updated_graph['nodes']:
        curr_idx = len([it for it, elem in enumerate(floor_none_ids) if elem < node['id']])
        new_id = node['id'] - curr_idx
        node_id_mapping[node['id']] = node['id'] - curr_idx
        node['id'] = new_id
        if node['class_name'] == 'towelrolled':
            node['class_name'] = 'towel'
        if node['prefab_name'] == 'Cabinet_2': 
            node['prefab_name'] = 'mH_FloorCabinet01'
            # print("HERE")

        # if node['prefab_name'] in translate_prefab:
        #     node['prefab_name'] = translate_prefab[node['prefab_name']]
    
    for edge in updated_graph['edges']:
        edge['from_id'] = node_id_mapping[edge['from_id']]
        edge['to_id'] = node_id_mapping[edge['to_id']]

    updated_graph['nodes'] = [node for node in updated_graph['nodes'] if node['class_name'] != 'kitchencabinets']
    updated_graph['nodes'] += [node for node in curr_graph['nodes'] if node['class_name'] == 'kitchencabinet']
    return updated_graph

def distance(a, b):
    return numpy.linalg.norm(np.array(a) - np.array(b))
def convert_task_goal(goals, old_graph, new_graph):
    id2node = {node['id']: node for node in old_graph['nodes']}
    new_goal = {}
    for char_id in goals.keys():
        converted_goal = {}
        for goal_name, count in goals[char_id].items():
            if 'on' in goal_name or 'inside' in goal_name or 'sit' in goal_name:
                class_name_old = id2node[int(goal_name.split('_')[-1])]['class_name']
                new_ids = [node for node in new_graph['nodes'] if node['class_name'] == class_name_old]
                if len(new_ids) != 1:
                    # whichever is close to coffeetable
                    node_coff = [node['bounding_box']['center'] for node in new_graph['nodes'] if node['class_name'] == 'coffeetable'][0]
                    dist = [(node, distance(node['bounding_box']['center'], node_coff)) for node in new_ids] 
                    it = np.argmin([x[1] for x in dist])
                    new_ids = [new_ids[it]]
                gsp = goal_name.split('_')
                goal_name_new = '_'.join([gsp[0], gsp[1], str(new_ids[0]['id'])])
            else:
                goal_name_new = goal_name
            converted_goal[goal_name_new] = count
        new_goal[char_id] = converted_goal
        
    return new_goal

if __name__ == '__main__':
    id2map = {}
    # file_input = '/Users/xavierpuig/Downloads/train_env_set_help_50_neurips.pik'
    file_input = '/Users/xavierpuig/Downloads/test_env_set_help_20_neurips.pik'

    comm = comm_unity.UnityCommunication()
    with open(file_input, 'rb') as f:
        file_in = pkl.load(f)
    for item in tqdm(range(len(file_in))):
        # ipdb.set_trace()
        env_id = file_in[item]['env_id']
        if env_id not in id2map:
            comm.reset(file_in[item]['env_id'])
            s, graph_sim = comm.environment_graph()
            id2map[env_id] = graph_sim
        graph_sim = copy.deepcopy(id2map[env_id])
        old_graph = copy.deepcopy(file_in[item]['init_graph'])
        file_in[item]['init_graph'] = remove_floor(file_in[item]['init_graph'], graph_sim)
        file_in[item]['task_goal'] = convert_task_goal(file_in[item]['task_goal'], old_graph, graph_sim)
    ipdb.set_trace()
    file_out = './dataset/test_env_set_help_20_neurips.pik'
    # file_out = './dataset/train_env_set_help_50_neurips.pik'
    
    with open(file_out, 'wb') as f:
        pkl.dump(file_in, f)
    ipdb.set_trace()