from utils import DictObjId
from gym import spaces, envs
import dgl
import numpy as np
import os
import json

class GraphHelper():
    def __init__(self):
        self.states = ['on', 'open', 'off', 'closed']
        self.relations = ['inside', 'close', 'facing', 'on']
        self.objects = self.get_objects()
        rooms = ['bathroom', 'bedroom', 'kitchen', 'livingroom']
        self.object_dict = DictObjId(self.objects + ['character'] + rooms + ['no_obj'])
        self.relation_dict = DictObjId(self.relations)
        self.state_dict = DictObjId(self.states)

        self.num_objects = 100
        self.num_edges = 200 
        self.num_edge_types = len(self.relation_dict)
        self.num_classes = len(self.object_dict)
        self.num_states = len(self.state_dict)

    def get_objects(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(f'{dir_path}/dataset/object_info.json', 'r') as f:
            content = json.load(f)
        objects = []
        for obj in content.values():
            objects += obj
        return objects
    
    def one_hot(self, states):
        one_hot = np.zeros(len(self.state_dict))
        for state in states:
            one_hot[self.state_dict.get_id(state)] = 1
        return one_hot

    def build_graph(self, graph, ids):
        ids += [node['id'] for node in graph['nodes'] if node['category'] == 'Rooms']
        ids += [node['id'] for node in graph['nodes'] if node['class_name'] == 'character']
        max_nodes = self.num_objects
        max_edges = self.num_edges
        edges = [edge for edge in graph['edges'] if edge['from_id'] in ids and edge['to_id'] in ids]
        nodes = [node for node in graph['nodes'] if node['id'] in ids]
        id2index = {node['id']: it for it, node in enumerate(nodes)}


        class_names = np.array([self.object_dict.get_id(node['class_name']) for node in nodes])
        node_states = np.array([self.one_hot(node['states']) for node in nodes])

        edge_types = np.array([self.relation_dict.get_id(edge['relation_type']) for edge in edges])

        if len(edges) > 0:
            edge_ids = np.concatenate(
                    [np.array([
                        id2index[edge['from_id']], 
                        id2index[edge['to_id']]])[None, :] for edge in edges], axis=0)

        else:
            pdb.set_trace()

        mask_edges = np.zeros(max_edges)
        all_edge_ids = np.zeros((max_edges, 2))
        all_edge_types = np.zeros((max_edges))

        mask_nodes = np.zeros((max_nodes))
        all_class_names = np.zeros((max_nodes))
        all_node_states = np.zeros((max_nodes, len(self.state_dict)))
        
        if len(edges) > 0:
            mask_edges[:len(edges)] = 1.
            all_edge_ids[:len(edges), :] = edge_ids
            all_edge_types[:len(edges)] = edge_types

        mask_nodes[:len(nodes)] = 1.
        all_class_names[:len(nodes)] = class_names
        all_node_states[:len(nodes)] = node_states
        return (all_class_names, all_node_states, 
                all_edge_ids, all_edge_types, mask_nodes, mask_edges)


class GraphSpace(spaces.Space):
    def __init__(self):
        self.shape = None
        self.dtype = "graph"

        pass
