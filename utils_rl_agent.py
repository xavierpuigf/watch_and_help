from utils import DictObjId
import torch
from gym import spaces, envs
from dgl import DGLGraph
import numpy as np
import os
import json
import pdb


class GraphHelper():
    def __init__(self):
        self.states = ['on', 'open', 'off', 'closed']
        self.relations = ['inside', 'close', 'facing', 'on']
        self.objects = self.get_objects()
        rooms = ['bathroom', 'bedroom', 'kitchen', 'livingroom']
        self.actions = [
            'turnleft',
            'walkforward',
            'turnright',
            'walktowards',
            'open',
            'close',
            'putback',
            'putin',
            'grab',
            'no_action'
        ]
        self.object_dict = DictObjId(self.objects + ['character'] + rooms + ['no_obj'])
        self.relation_dict = DictObjId(self.relations)
        self.state_dict = DictObjId(self.states)
        self.action_dict = DictObjId(self.actions, include_other=False)

        self.num_objects = 100
        self.num_edges = 200 
        self.num_edge_types = len(self.relation_dict)
        self.num_classes = len(self.object_dict)
        self.num_states = len(self.state_dict)


        
        self.obj1_affordance = None
        self.obj2_affordance = None
        self.get_action_affordance_map()

    def update_probs(self, log_probs, i, actions, object_classes):
        """
        :param log_probs: current log probs
        :param i: which action are we currently considering
        :param actions: actions already selected
        :return:
        """
        inf_val = 1e9
        if i == 1:
            # Deciding on the object
            return log_probs
        elif i == 0:
            # Deciding on the action
            selected_obj1 = object_classes[range(object_classes.shape[0]), actions[1]].long()
            mask = torch.Tensor(self.obj1_affordance[None, :][:, :, selected_obj1] == 1).to(log_probs.device)
            log_probs = log_probs * mask + (1.-mask) * -inf_val
            return log_probs

        else:
            # deciding on object 2
            selected_action = actions[0]

            # batch x object_class
            mask_object_class = torch.Tensor(self.obj2_affordance[None, :][:, selected_action, :] == 1).unsqueeze(1).to(object_classes.device)

            # batch x nodes x object_class
            one_hot = torch.LongTensor(object_classes.shape[0], object_classes.shape[1], mask_object_class.shape[-1]).zero_().to(object_classes.device)
            target_one_hot = one_hot.scatter_(2, object_classes.unsqueeze(-1).long(), 1)

            # the first node is the character
            mask_nodes = ((mask_object_class * target_one_hot).sum(-1) > 0)[:, :log_probs.shape[1]]
            mask = mask_nodes.to(log_probs.device).float()
            log_probs = log_probs * mask + (1.-mask) * -inf_val


            return log_probs


    def get_action_affordance_map(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/dataset/object_info.json', 'r') as f:
            content = json.load(f)

        n_actions = len(self.actions)
        n_objects = len(self.object_dict)
        self.obj1_affordance = np.zeros((n_actions, n_objects))
        self.obj2_affordance = np.zeros((n_actions, n_objects))

        id_no_obj = self.object_dict.get_id('no_obj')
        id_grab = np.array([self.object_dict.get_id(obj_name) for obj_name in content['objects_grab']])
        id_surface = np.array([self.object_dict.get_id(obj_name) for obj_name in content['objects_surface']])
        id_containers = np.array([self.object_dict.get_id(obj_name) for obj_name in content['objects_inside']])
        for action in ['no_action', 'turnleft', 'walkforward', 'turnright']:
            action_id = self.action_dict.get_id(action)
            self.obj1_affordance[action_id, id_no_obj] = 1
            self.obj2_affordance[action_id, id_no_obj] = 1

        for action in ['walktowards', 'open', 'close', 'grab']:
            action_id = self.action_dict.get_id(action)
            self.obj2_affordance[action_id, id_no_obj] = 1
            if action in ['open', 'close']:
                self.obj1_affordance[action_id, id_containers] = 1
            if action in ['grab']:
                self.obj1_affordance[action_id, id_grab] = 1
            if action in ['walktowards']:
                self.obj1_affordance[action_id, :] = 1
                self.obj1_affordance[action_id,id_no_obj] = 0
                
        for action in ['putback', 'putin']:
            self.obj1_affordance[action_id, id_grab] = 1
            id2 = id_containers if action == 'putin' else id_surface
            self.obj2_affordance[action_id, id2] = 1







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

    def build_graph(self, graph, ids, plot_graph=False):
        ids = [node['id'] for node in graph['nodes'] if node['category'] == 'Rooms'] + ids
        id2node = {node['id']: node for node in graph['nodes']}
        # Character is always the first one
        ids = [node['id'] for node in graph['nodes'] if node['class_name'] == 'character'] + ids
        max_nodes = self.num_objects
        max_edges = self.num_edges
        edges = [edge for edge in graph['edges'] if edge['from_id'] in ids and edge['to_id'] in ids]
        nodes = [id2node[idi] for idi in ids]
        nodes.append({'id': -1, 'class_name': 'no_obj', 'states': []})

        id2index = {node['id']: it for it, node in enumerate(nodes)}

        class_names_str = [node['class_name'] for node in nodes]
        visible_nodes = [(node['class_name'], node['id']) for node in nodes]

        class_names = np.array([self.object_dict.get_id(class_name) for class_name in class_names_str])
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
        all_class_names = np.zeros((max_nodes)).astype(np.int32)
        all_node_states = np.zeros((max_nodes, len(self.state_dict)))
        
        if len(edges) > 0:
            mask_edges[:len(edges)] = 1.
            all_edge_ids[:len(edges), :] = edge_ids
            all_edge_types[:len(edges)] = edge_types

        mask_nodes[:len(nodes)] = 1.
        all_class_names[:len(nodes)] = class_names
        all_node_states[:len(nodes)] = node_states

        
        if plot_graph:
            graph_viz = DGLGraph()
            graph_viz.add_nodes(len(nodes), {'names': class_names})
            labeldict =  {it: class_str for it, class_str in enumerate(class_names_str)}
        else:
            labeldict = None
            graph_viz = None

        return (all_class_names, all_node_states, 
                all_edge_ids, all_edge_types, mask_nodes, mask_edges), (graph_viz, labeldict, visible_nodes)

def can_perform_action(action, o1, o2, agent_id, graph):
    num_args = len([None for ob in [o1, o2] if ob is not None])
    grabbed_objects = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id and edge['relation_type'] in ['HOLDS_RH', 'HOLD_LH']]
    if num_args != args_per_action(action):
        return False
    if 'put' in action:
        if o1 not in grabbed_objects:
            return False

    return True

def args_per_action(action):

    action_dict = {'turnleft': 0,
    'walkforward': 0,
    'turnright': 0,
    'walktowards': 1,
    'open': 1,
    'close': 1,
    'putback':2,
    'putin': 2,
    'grab': 1,
    'no_action': 0}
    return action_dict[action]

class GraphSpace(spaces.Space):
    def __init__(self):
        self.shape = None
        self.dtype = "graph"

        pass
