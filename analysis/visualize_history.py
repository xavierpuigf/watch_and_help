import json
import pickle as pkl
import cv2
import numpy as np
import scipy.special
import pdb
import sys
sys.path.append('../')
import utils_viz
import time
sys.path.append('../../../virtualhome/simulation')
from unity_simulator.comm_unity import UnityCommunication

cam_id = 74
def add_character(comm, character_resource='Chars/Male1'):
    response = comm.post_command(
        {'id': str(time.time()), 'action': 'add_character', 
         'stringParams':[json.dumps({'character_resource': character_resource})]})

    return response['success'], response['message']

with open('class_name_equivalence_extended.json', 'r') as f:
    name_equivalence = json.load(f)

def update_graph(new_graph, graph, char_id):
    nodes = new_graph['nodes']
    edges = new_graph['edges']

    missing_nodes = []
    for node in nodes:
        name = node['class_name']
        if name not in name_equivalence:
            missing_nodes.append(node['id'])

    # Replace character id
    original_id = -1
    for i in range(len(nodes)):
        if nodes[i]['class_name'] == 'character':
            # nodes[i] = char.copy()
            original_id = nodes[i]['id']
            nodes[i]['id'] = char_id

    for edge in edges:
        if edge['from_id'] == original_id:
            edge['from_id'] = char_id
        if edge['to_id'] == original_id:
            edge['to_id'] = char_id             


    new_graph['nodes'] = list(filter(lambda item: item['id'] not in missing_nodes, nodes))
    new_graph['edges'] = list(filter(lambda item: (item['from_id'] not in missing_nodes) and 
                                     (item['to_id'] not in missing_nodes), edges))

    ids_new_graph = [node['id'] for node in new_graph['nodes'] ]
    ids_reserved = [node['id'] for node in graph['nodes'] if 
                    node['id'] not in ids_new_graph and 
                    node['class_name'] in ['ceiling', 'floor', 'door', 'wall', 'walllamp', 'ceilinglamp'] ]

    nodes_reserved = [node for node in graph['nodes'] if 
                      node['id'] in ids_reserved]
    new_graph['nodes'] += nodes_reserved

    edges_reserved = [edge for edge in graph['edges'] if (
                      edge['from_id'] in ids_reserved or edge['to_id'] in ids_reserved) and 
                      edge['relation_type'] == 'INSIDE' and 
                      edge['from_id'] in new_graph['nodes'] and 
                      edge['to_id'] in new_graph['nodes']]
    new_graph['edges'] += edges_reserved

def visualize_plan(comm, info, plan_id, time):
    print('Loading image')
    success, camera = comm.camera_data([cam_id])
    success, image = comm.camera_image([cam_id], image_height=960, image_width=1280)
    print('Image received')
    camera = camera[0]
    image = image[0]
    _, graph = comm.environment_graph()
    id2node = {}
    for node in graph['nodes']:
        id2node[node['id']] = node

    belief = info['belief'][time][2007]['INSIDE']
    prob = []
    object_coords = []
    softmax_bel = scipy.special.softmax(belief[1])
    print(softmax_bel)
    names = []
    for it, idi in enumerate(belief[0]):
        if idi is not None:
            print(idi)
            coord_idi = [(x['bounding_box']['center']) for x in graph['nodes'] if x['id'] == idi][0]
            names.append([x['class_name'] for x in graph['nodes'] if x['id'] == idi][0])
            object_coords.append(np.array(coord_idi)[:, None])
            prob.append(softmax_bel[it])
    object_coords = np.concatenate(object_coords, 1)
    prob = np.array(prob)
    img = utils_viz.viz_belief(image, camera, object_coords, prob)
    return (img*255.).astype(np.uint8)

file_example = 'TrimmedTestScene2_graph_1.json'
original_graph = '../dataset_toy4/init_envs/{}'.format(file_example)
with open(original_graph, 'r') as f:
    graph_modif = json.load(f)
    graph_modif = graph_modif['init_graph']

#char_node_modif = [x['id'] for x in graph_modif['nodes'] if x['class_name'] == 'character'][0]
#new_room_modif = [x['to_id'] for x in graph_modif['edges'] if x['from_id'] == char_node_modif and x['relation_type'] == 'INSIDE'][0]
comm = UnityCommunication()
comm.reset(1)
add_character(comm)
_ = comm.camera_image([cam_id])
s, graph = comm.environment_graph()
char_node = [x['id'] for x in graph['nodes'] if x['class_name'] == 'character'][0]
#graph['edges'] = [x for x in graph['edges'] if x['from_id'] != char_node and x['to_id'] != char_node]
#graph['edges'].append({'from_id': char_node, 'relation_type': 'INSIDE', 'to_id': new_room_modif})
#resp = comm.expand_scene(graph)
chars = [node for node in graph['nodes'] if node['class_name'] == 'character']
char = chars[0]
char_id = char['id']
update_graph(graph_modif, graph, char_id)
resp = comm.expand_scene(graph_modif)
_ = comm.camera_image([cam_id])
with open('../logdir/history.pik', 'rb') as f:
    info = pkl.load(f)
pdb.set_trace()
if True: #resp[0]:
    print('Expanded')
    num_steps = len(info['belief'])
    print(num_steps)
    pdb.set_trace()
    for time in range(num_steps):
        img = visualize_plan(comm, info, 0, time)
        action = info['action']
        print(action[time])
        aux = comm.render_script(['<char0> '+action[time]], gen_vid=False, image_synthesis=['normal'])
        print(aux)
        cv2.imwrite('res_{:02d}.png'.format(time), img)
    s, img = comm.camera_image([cam_id])
    cv2.imwrite('res_{:02d}.png'.format(num_steps), img[0])

