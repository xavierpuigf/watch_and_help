from utils import DictObjId
import torch
from gym import spaces, envs
from dgl import DGLGraph
import numpy as np
import os
import json
import pdb


class GraphHelper():
    def __init__(self, max_num_objects=100, max_num_edges=200, current_task=None, simulator_type='unity'):
        self.states = ['on', 'open', 'off', 'closed']
        self.relations = ['inside', 'close', 'facing', 'on']
        self.simulaor_type = simulator_type
        self.objects = self.get_objects()
        self.rooms = ['bathroom', 'bedroom', 'kitchen', 'livingroom']

        if simulator_type == 'unity':
            self.actions = [
                'turnleft',
                'walkforward',
                'turnright',
                'walktowards',
                'open',
                'close',
                'put',
                'grab',
                'no_action'
            ]
        else:
            self.actions = [
                'walk',
                'open',
                'close',
                'put',
                'grab',
                'no_action'
            ]
        self.object_dict = DictObjId(self.objects + ['character'] + self.rooms + ['no_obj'])
        self.relation_dict = DictObjId(self.relations)
        self.state_dict = DictObjId(self.states)
        self.action_dict = DictObjId(self.actions, include_other=False)

        self.num_objects = max_num_objects
        self.num_edges = max_num_edges
        self.num_edge_types = len(self.relation_dict)
        self.num_classes = len(self.object_dict)
        self.num_states = len(self.state_dict)


        
        self.obj1_affordance = None
        self.get_action_affordance_map(current_task=current_task)




    def get_action_affordance_map(self, current_task=None, id2node=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/dataset/object_info_small.json', 'r') as f:
            content = json.load(f)

        n_actions = len(self.actions)
        n_objects = len(self.object_dict)
        self.obj1_affordance = np.zeros((n_actions, n_objects))

        id_no_obj = self.object_dict.get_id('no_obj')
        id_grab = np.array([self.object_dict.get_id(obj_name) for obj_name in content['objects_grab']])
        id_surface = np.array([self.object_dict.get_id(obj_name) for obj_name in content['objects_surface']])
        id_containers = np.array([self.object_dict.get_id(obj_name) for obj_name in content['objects_inside']])
        for action in self.actions:
            action_id = self.action_dict.get_id(action)
            if args_per_action(action) == 0:

                self.obj1_affordance[action_id, id_no_obj] = 1

            if args_per_action(action) == 1:

                if action in ['open', 'close']:
                    self.obj1_affordance[action_id, id_containers] = 1

                elif action in ['grab']:
                    self.obj1_affordance[action_id, id_grab] = 1

                    if current_task is not None:
                        self.obj1_affordance[action_id, :] = 0
                        obj_names = [t.split('_')[1] for t in current_task[0].keys()]
                        ids_goal = [self.object_dict.get_id(obj_name) for obj_name in obj_names]
                        id_goal = np.array(ids_goal)
                        self.obj1_affordance[action_id, id_goal] = 1


                elif action in ['walktowards', 'walk']:
                    self.obj1_affordance[action_id, :] = 1
                    self.obj1_affordance[action_id, id_no_obj] = 0



                # putin, put
                elif action.startswith('put'):
                    id2 = id_containers if action == 'putin' else id_surface
                    self.obj1_affordance[action_id, id2] = 1

                    if current_task is not None:
                        self.obj1_affordance[action_id, :] = 0
                        obj_names2 = [id2node[int(t.split('_')[2])]['class_name'] for t in
                                      current_task[0].keys() if t.split('_')[0] not in ['holds', 'sit', 'turnOn']]
                        ids_goal2 = np.array([self.object_dict.get_id(obj_name) for obj_name in obj_names2])
                        self.obj1_affordance[action_id, ids_goal2] = 1
        # self.obj1_affordance[:,self.object_dict.get_id('kitchencounterdrawer')] = 0
        self.obj1_affordance[self.action_dict.get_id('open'),self.object_dict.get_id('kitchencounterdrawer')] = 0
        self.obj1_affordance[self.action_dict.get_id('close'),self.object_dict.get_id('kitchencounterdrawer')] = 0
        self.obj1_affordance[self.action_dict.get_id('walktowards'),self.object_dict.get_id('kitchencounterdrawer')] = 0
        self.obj1_affordance[self.action_dict.get_id('walktowards'),self.object_dict.get_id('character')] = 0
        self.obj1_affordance[:,id_no_obj] = 0


    def get_objects(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(f'{dir_path}/dataset/object_info_small.json', 'r') as f:
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

    def build_graph(self, graph, character_id, ids=None, plot_graph=False, level=1):
        if ids is None:
            ids = [node['id'] for node in graph['nodes']]

        for node in graph['nodes']:
            if node['category'] == 'Rooms':
                assert(node['class_name'] in self.rooms)

        if level > 0:
            ids = [node['id'] for node in graph['nodes'] if node['category'] == 'Rooms'] + ids
        ids = [idi for idi in ids if idi != character_id]
        ids = list(set(ids))
        id2node = {node['id']: node for node in graph['nodes']}


        # Character is always the first one
        ids = [character_id] + ids
        max_nodes = self.num_objects
        max_edges = self.num_edges
        edges = [edge for edge in graph['edges'] if edge['from_id'] in ids and edge['to_id'] in ids]
        nodes = [id2node[idi] for idi in ids]
        nodes.append({'id': -1, 'class_name': 'no_obj', 'states': []})

        char_coord = np.array(nodes[0]['bounding_box']['center'])
        rel_coords = [np.array([0,0,0])[None, :] if 'bounding_box' not in node.keys() else (np.array(node['bounding_box']['center']) - char_coord)[None, :] for node in nodes]

        id2index = {node['id']: it for it, node in enumerate(nodes)}

        class_names_str = [node['class_name'] for node in nodes]
        visible_nodes = [(node['class_name'], node['id']) for node in nodes]
        node_ids = [node['id'] for node in nodes]

        class_names = np.array([self.object_dict.get_id(class_name) for class_name in class_names_str])
        node_states = np.array([self.one_hot(node['states']) for node in nodes])

        edge_types = np.array([self.relation_dict.get_id(edge['relation_type']) for edge in edges])

        if len(edges) > 0:
            edge_ids = np.concatenate(
                    [np.array([
                        id2index[edge['from_id']], 
                        id2index[edge['to_id']]])[None, :] for edge in edges], axis=0)

        # else:
        #     pdb.set_trace()

        mask_edges = np.zeros(max_edges)
        all_edge_ids = np.zeros((max_edges, 2))
        all_edge_types = np.zeros((max_edges))

        mask_nodes = np.zeros((max_nodes))
        all_class_names = np.zeros((max_nodes)).astype(np.int32)
        all_node_states = np.zeros((max_nodes, len(self.state_dict)))
        all_node_ids = np.zeros((max_nodes)).astype(np.int32)

        if len(edges) > 0:
            mask_edges[:len(edges)] = 1.
            all_edge_ids[:len(edges), :] = edge_ids
            all_edge_types[:len(edges)] = edge_types

        mask_nodes[:len(nodes)] = 1.
        all_class_names[:len(nodes)] = class_names
        all_node_states[:len(nodes)] = node_states
        all_node_ids[:len(nodes)] = node_ids

        obj_coords = np.zeros((max_nodes, 3))
        obj_coords[:len(nodes)] = np.concatenate(rel_coords, 0)
        
        if plot_graph:
            graph_viz = DGLGraph()
            graph_viz.add_nodes(len(nodes), {'names': class_names})
            labeldict =  {it: class_str for it, class_str in enumerate(class_names_str)}
        else:
            labeldict = None
            graph_viz = None
        #print("CLASSNAMES") 
        #print(all_class_names[:5])
        output = {
            'class_objects': all_class_names,
            'states_objects': all_node_states,
            'edge_tuples': all_edge_ids,
            'edge_classes': all_edge_types,
            'mask_object': mask_nodes,
            'mask_edge': mask_edges,
            'object_coords': obj_coords,
            'node_ids': all_node_ids
        }
        #print(node_ids[:len(nodes)])
        return output, (graph_viz, labeldict, visible_nodes)

def can_perform_action(action, o1, o1_id, agent_id, graph):
    if action == 'no_action':
        return None
    # if action in ['open', 'close', 'grab', 'putback']:
    #     return False
    print('Attemptinf', action, o1)
    obj2_str = ''
    obj1_str = ''
    id2node = {node['id']: node for node in graph['nodes']}
    num_args = 0 if o1 is None else 1
    grabbed_objects = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id and edge['relation_type'] in ['HOLDS_RH', 'HOLD_LH']]
    if num_args != args_per_action(action):
        return None
    
    # if 'walk' not in action and 'turn' not in action:
    #     return None
    close_edge = len([edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id and edge['to_id'] == o1_id and edge['relation_type'] == 'CLOSE']) > 0
    if action == 'grab':
        print(agent_id, o1_id, close_edge)

    if (action in ['grab', 'open', 'close']) and not close_edge:
        return None

    if action == 'open':
        print(o1_id, id2node[o1_id]['states'])
        if 'OPEN' in id2node[o1_id]['states'] or 'CLOSED' not in id2node[o1_id]['states']:
            return None

    if action == 'close':
        print(o1_id, id2node[o1_id]['states'])
        if 'CLOSED' in id2node[o1_id]['states'] or 'OPEN' not in id2node[o1_id]['states']:
            return None

    if 'put' in action:
        if len(grabbed_objects) == 0:
            return None
        else:
            o2_id = grabbed_objects[0]
            if o2_id == o1_id:
                return None
            o2 = id2node[o2_id]['class_name']
            obj2_str = f'<{o2}> ({o2_id})'

    if o1 is not None:
        obj1_str = f'<{o1}> ({o1_id})'
    if o1_id in id2node.keys():
        if id2node[o1_id]['class_name'] == 'character':
            return None
    if action.startswith('put'):

        if 'CONTAINERS' in id2node[o1_id]['properties']:
            action = 'putin'
        elif 'SURFACES' in id2node[o1_id]['properties']:
            action = 'putback'
    if action.startswith('walk'):
        action = 'walkto'
    action_str = f'[{action}] {obj2_str} {obj1_str}'.strip()
    #print(action_str)
    print(action_str)
    return action_str

def args_per_action(action):

    action_dict = {'turnleft': 0,
    'walkforward': 0,
    'turnright': 0,
    'walktowards': 1,
    'open': 1,
    'close': 1,
    'putback':1,
    'putin': 1,
    'put': 1,
    'grab': 1,
    'no_action': 0,
    'walk': 1}
    return action_dict[action]

class GraphSpace(spaces.Space):
    def __init__(self):
        self.shape = None
        self.dtype = "graph"

        pass

def update_probs(log_probs, i, actions, object_classes, mask_observations, obj1_affordance):
    """
    :param log_probs: current log probs
    :param i: which action are we currently considering
    :param actions: actions already selected
    :param mask_observations: bs x max_nodes with the valid nodes
    :return:
    """
    #pdb.set_trace()
    inf_val = 1e9
    if i == 1:
        # Deciding on the object
        log_probs =  log_probs * mask_observations + (1.-mask_observations) * -inf_val
        # check if an object cannot in no class

        # b x num_classes
        #mask_object_class = obj1_affordance.sum(1) > 0
        #if np.sum(mask_object_class) != mask_object_class.shape[0]:
        #    pdb.set_trace()
        ## batch x nodes x object_class
        #one_hot = torch.LongTensor(object_classes.shape[0], object_classes.shape[1],
        #                           mask_object_class.shape[-1]).zero_().to(object_classes.device)
        #target_one_hot = one_hot.scatter_(2, object_classes.unsqueeze(-1).long(), 1)
        #mask_nodes = ((mask_object_class * target_one_hot).sum(-1) > 0)[:, :log_probs.shape[1]]
        #mask = mask_nodes.to(log_probs.device).float()
        #log_probs = log_probs * mask + (1. - mask) * -inf_val
        return log_probs

    elif i == 0:
        # Deciding on the action
        selected_obj1 = torch.gather(object_classes, 1, actions[1].long())

        mask = torch.gather(obj1_affordance, 2, selected_obj1.unsqueeze(-2).repeat(1, obj1_affordance.shape[1], 1).long()).squeeze(-1).float().to(log_probs.device)
        # mask[action_dict.get_id('open'),object_dict.get_id('kitchencounterdrawer') ]= 0
        log_probs = log_probs * mask + (1.-mask) * -inf_val
        
        #print("CLASS OBJ")
        #print(object_classes[0, :5])
        #print(actions, 'CLASS', selected_obj1)
        #print(log_probs)
        #if log_probs[:, 8] > -inf_val:
        #    print(log_probs)
        #   pdb.set_trace()

        return log_probs
