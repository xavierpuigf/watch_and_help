import graphviz
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import copy
import scipy.special
import json
import pdb
import numpy as np
import random

def getclass(node):
    return '{}\n{}'.format(node['class_name'], node['id'])

def delete_redundant_edges_and_ids(graph):
    class_nodes_delete = ['wall', 'floor', 'ceiling', 'door', 'curtain', 'window', 'doorjamb']
    ids_delete = [x['id'] for x in graph['nodes'] if x['class_name'] in class_nodes_delete]

    graph['nodes'] = [x for x in graph['nodes'] if x['id'] not in ids_delete]
    graph['edges'] = [x for x in graph['edges'] if x['from_id'] not in ids_delete and x['to_id'] not in ids_delete]

    parent_node = {}
    children_node = {}
    for it, edge in enumerate(graph['edges']):
        if edge['relation_type'] == 'INSIDE':
            if edge['to_id'] not in parent_node.keys(): parent_node[edge['to_id']] = []
            parent_node[edge['to_id']].append(edge['from_id'])
            if edge['from_id'] not in children_node.keys(): children_node[edge['from_id']] = []
            children_node[edge['from_id']].append(edge['to_id'])

    final_edges = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            all_parents = children_node[edge['from_id']]
            all_children = parent_node[edge['to_id']]
            if len(set(all_parents).intersection(all_children)) > 0:
                continue
        final_edges.append(edge)
    graph['edges'] = final_edges
    return graph

def graph2im(graph, special_nodes={}):
    """
    Outputs an image given a graph
    :param graph:
    :return:
    """
    # graph = delete_redundant_edges_and_ids(graph)

    class_nodes_delete = ['wall', 'floor', 'ceiling', 'curtain']
    categories_delete = ['Doors']
    ids_delete = [x['id'] for x in graph['nodes'] if x['class_name'] in class_nodes_delete or x['category'] in categories_delete]

    nodes = [x for x in graph['nodes'] if x not in ids_delete]
    edges = [x for x in graph['edges'] if x['from_id'] not in ids_delete and x['to_id'] not in ids_delete]

    id2node = {x['id']: x for x in nodes}

    g = graphviz.Digraph(engine='dot')
    g.attr(compound='true')

    # g.attr(rank='min')

    container_nodes = list(set([x['to_id'] for x in edges if x['relation_type'] == 'INSIDE']))
    children = {}
    parent = {}
    num_children_subgraph = {}
    for edge in edges:
        if edge['relation_type'] == 'INSIDE':
            if edge['to_id'] not in children:
                children[edge['to_id']] = []
                num_children_subgraph[edge['to_id']] = 0
            parent[edge['from_id']] = edge['to_id']
            children[edge['to_id']].append(edge['from_id'])
            if edge['from_id'] in container_nodes:
                num_children_subgraph[edge['to_id']] += 1

    subgraphs_added = {}
    
    curr_subgraphs = [x for (x,y) in num_children_subgraph.items() if y == 0]
    # pdb.set_trace()
    while len(curr_subgraphs) > 0:
        next_subgraphs = []
        for curr_subgraph_id in curr_subgraphs:
            try:
                name_graph = getclass(id2node[curr_subgraph_id])
            except:
                curr_subgraph_id
            cng = graphviz.Digraph(name='cluster_'+str(curr_subgraph_id))
            cng.attr(label=name_graph)
            cng.node(name=str(curr_subgraph_id), style='invis')
            # pdb.set_trace()
            children_c_graph = children[curr_subgraph_id]
            for child in children_c_graph:
                if child in subgraphs_added:
                    cng.subgraph(subgraphs_added[child])

                else:
                    color_special = {
                        'agent': 'darkseagreen',
                        'goal': 'lightblue'
                    }
                    if child in special_nodes:
                        color_sp = color_special[special_nodes[child]]
                        cng.node(name=str(child), label=getclass(id2node[child]), color=color_sp, style='filled')

                    else:
                        cng.node(name=str(child), label=getclass(id2node[child]))

            subgraphs_added[curr_subgraph_id] = cng
            if curr_subgraph_id == 1:
                pdb.set_trace()
            if curr_subgraph_id not in parent:
                g.subgraph(cng)
                continue
            parent_graph = parent[curr_subgraph_id]
            num_children_subgraph[parent_graph] -= 1
            if num_children_subgraph[parent_graph] == 0:
                next_subgraphs.append(parent_graph)
        curr_subgraphs = next_subgraphs

    colors = {
        'INSIDE': 'yellow',
        'ON': 'blue',
        'CLOSE': 'purple',
        'CLOSE_CHAR': 'orange',
        'FAKE_CLOSE': 'orange',
        'FACING': 'red',
        'BETWEEN': 'green'

    }
    style = {
        'INSIDE': '',
        'ON': '',
        'CLOSE': 'invis',
        'FACING': 'invis',
        'BETWEEN': 'invis',
        'FAKE_CLOSE': 'invis',
        'CLOSE_CHAR': ''

    }
    
    # If ther are no close edges, add close from char to objects in the room
    close_edges = [x for x in edges if 'CLOSE' in x['relation_type']]
    extra_edges = []
    if len(close_edges) == 0:
        id_char = [x['id'] for x in nodes if x['class_name'] == 'character'][0]
        nodes_same_room = [x['id'] for x in nodes if x['id'] in parent.keys() and parent[x['id']] == parent[id_char] and id_char != x['id']]
        for node_id in nodes_same_room:
            extra_edges.append({'from_id': id_char, 'to_id': node_id, 'relation_type': 'FAKE_CLOSE'})

    max_num = 0.2
    id_char = [x for x,y in special_nodes.items() if y == 'agent']
    if len(id_char) > 0:
        id_char = id_char[0]
    for edge in edges+extra_edges:
        rt = edge['relation_type']
        if rt != 'INSIDE' and edge['from_id'] not in ids_delete and edge['to_id'] not in ids_delete:
            if rt == 'CLOSE':
                # print(edge['from_id'], edge['to_id'], edge['relation_type'])
                if edge['from_id'] == id_char:
                    rt = 'CLOSE_CHAR'
                    print(rt)
                else:
                    if parent[edge['from_id']] != parent[edge['to_id']]:
                        continue
                    if random.random() < max_num:
                        continue
            # if ((edge['to_id'] in parent and parent[edge['to_id']] == edge['from_id']) or
            #    (edge['from_id'] in parent and parent[edge['from_id']] == edge['to_id'])):
            #     continue

            if edge['from_id'] not in children and edge['to_id'] not in children:
                g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt], style=style[rt])

            elif edge['from_id'] not in children:
                g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                       lhead='cluster_' + str(edge['to_id']), style=style[rt])
            elif edge['to_id'] not in children:
                g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                       ltail='cluster_' + str(edge['from_id']), style=style[rt])
            else:
                g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                       ltail='cluster_'+str(edge['from_id']), lhead='cluster_'+str(edge['to_id']), style=style[rt])
    return g


def belief2im(belief, special_nodes={}):
    """
    Outputs an image given a graph
    :param graph:
    :return:
    """

    # graph = delete_redundant_edges_and_ids(graph)
    graph = belief.sampled_graph
    class_nodes_delete = ['wall', 'floor', 'ceiling', 'door']
    ids_delete = [x['id'] for x in graph['nodes'] if x['class_name'] in class_nodes_delete]

    nodes = [x for x in graph['nodes'] if x not in ids_delete]
    edges = [x for x in graph['edges'] if x['from_id'] not in ids_delete and x['to_id'] not in ids_delete]

    id2node = {x['id']: x for x in nodes} 

    g = graphviz.Digraph(engine='fdp')

    for node in nodes:
        g.node(name=str(node['id']), label=getclass(node))

    colors = {
        'INSIDE': 'green',
        'ON': 'blue',
        'CLOSE': 'purple',
        'CLOSE_CHAR': 'orange',
        'FACING': 'red',
        'BETWEEN': 'green'

    }

    ids_used = []
    for from_id in belief.edge_belief.keys():
        for relation in ['ON', 'INSIDE']:
            value_belief = belief.edge_belief[from_id][relation]
            for to_id, belief_val in zip(*value_belief):
                if to_id is not None:
                    if relation == 'INSIDE':
                        ids_used.append(from_id)
                    if belief_val <= 0.:
                        continue
                    g.edge(str(from_id), str(to_id), color=colors[relation], 
                           arrowtail=str(belief_val),
                           penwidth=str(belief_val), weight=str(belief_val))

    for id in belief.room_node.keys():
        #pdb.set_trace()
        for belief_id, belief_val in zip(*belief.room_node[id]):
            if id not in ids_used:
                g.edge(str(id), str(belief_id), color=colors['INSIDE'],
                           arrowtail=str(belief_val),
                           penwidth=str(belief_val), weight=str(belief_val))
    return g



def print_graph(graph, output='graph_example.gv'):
    id_char = [x['id'] for x in graph['nodes'] if x['class_name'] == 'character'][0]
    g = graph2im(graph)
    g.render(output)

def print_belief(belief, output='belief_example.gv'):
    id_char = [x['id'] for x in belief.sampled_graph['nodes'] if x['class_name'] == 'character'][0]
    g = belief2im(belief, id_char)
    g.render(output)


def world2im(camera_data, wcoords):
    proj = np.array(camera_data['projection_matrix']).reshape((4,4)).transpose()
    w2cam = np.array(camera_data['world_to_camera_matrix']).reshape((4,4)).transpose()
    cw = np.concatenate([wcoords, np.ones((1, wcoords.shape[1]))], 0) # 4 x N
    pixelcoords = np.matmul(proj, np.matmul(w2cam, cw)) # 4 x N
    pixelcoords = pixelcoords/pixelcoords[-1, :]
    pixelcoords = (pixelcoords + 1)/2.
    pixelcoords[1,:] = 1. - pixelcoords[1, :]
    return pixelcoords[:2, :]

def obtain_hmap(pixel_coords, probabilities, img):
    # Given some pixel coordinates and probabilities, get a heatmap given some gaussians
    sigma = 20 # What is the extension of the heatmap
    norm = 2*np.pi*(sigma**2)
    values = np.zeros(img.shape[:2])
    pixel_coords = pixel_coords.astype(np.int32)
    indices_valid = np.logical_and(pixel_coords[1,:] < values.shape[0], 
                                   pixel_coords[1,:] >= 0)
    indices_valid2 = np.logical_and(pixel_coords[0,:] < values.shape[1], 
                                   pixel_coords[0,:] >= 0)
    indices_valid = np.array(np.logical_and(indices_valid, indices_valid2))
    pixel_coords = pixel_coords[:, indices_valid]
    probabilities = probabilities[indices_valid]
    
    values[pixel_coords[1, :], pixel_coords[0, :]] = probabilities
    result = gaussian_filter(values*norm, sigma, mode='nearest')
    #print(result.shape)
    img_heat_map = blend(img, result)
    return img_heat_map

def viz_belief(image, camera_data, object_coords, prob):
    imgcoords = world2im(camera_data, object_coords)
    imgcoords[0,:]*= image.shape[1]
    imgcoords[1,:]*= image.shape[0]
    res = obtain_hmap(imgcoords, prob, image)
    return res

def blend(img, heatmap):
    cmap = plt.cm.jet
    heatmap = np.clip(heatmap, 0, 1.)
    colorheatmap = cmap(1-heatmap)
    contrast = 3.
    outImage = img/255. * (1-contrast*heatmap[:, :, None]) + (contrast*heatmap[:, :, None])*colorheatmap[:,:,:3]
    
    return outImage




if __name__ == '__main__':
    input_graph = 'dataset_toy3/init_envs/TrimmedTestScene5_graph_8.json'
    with open(input_graph, 'r') as f:
        graph = json.load(f)
    g = graph2im(graph['init_graph'])
    g.render('graph_example.gv')
