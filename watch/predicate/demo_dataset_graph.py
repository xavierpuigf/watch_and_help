import os
import random
import copy
import json
import re
#import imageio
import numpy as np
from termcolor import colored
from glob import glob
from PIL import Image
import ipdb
import pickle
import pdb

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms


################################
# Demonstration
################################

def get_dataset(args, train):
    train_data, test_data, new_test_data, action_predicates, all_action, all_object, goal_objects, goal_targets, goal_predicates, graph_class_names, graph_node_states, max_goal_length, max_action_length, max_node_length, max_subgoal_length = gather_data(args)
    train_dset = demo_dset(args, train_data, action_predicates, all_action, all_object, goal_objects, goal_targets, goal_predicates, graph_class_names, graph_node_states, max_goal_length, max_action_length, max_node_length, max_subgoal_length)
    test_dset = demo_dset(args, test_data, action_predicates, all_action, all_object, goal_objects, goal_targets, goal_predicates, graph_class_names, graph_node_states, max_goal_length, max_action_length, max_node_length, max_subgoal_length)
    new_test_dset = demo_dset(args, new_test_data, action_predicates, all_action, all_object, goal_objects, goal_targets, goal_predicates, graph_class_names, graph_node_states, max_goal_length, max_action_length, max_node_length, max_subgoal_length)
    return train_dset, test_dset, new_test_dset


def collate_fn(data_list):
    graph_data = [data[0] for data in data_list]
    batch_goal_index = [data[1] for data in data_list]
    batch_valid_action_with_walk_index = [data[2] for data in data_list]


    if len(graph_data[0])==3:
        batch_graph_length = [d[0] for d in graph_data]
        batch_graph_input = [d[1] for d in graph_data]
        batch_file_name = [d[2] for d in graph_data]
    else:
        batch_graph_length = [d[0] for d in graph_data]
        batch_file_name = [d[1] for d in graph_data]

    if len(graph_data[0])==3:
        batch_demo_data = (
            np.arange(len(batch_graph_length)), 
            batch_graph_length, 
            batch_graph_input,
            batch_file_name
        )
    else:
        batch_demo_data = (
            np.arange(len(batch_graph_length)), 
            batch_graph_length,
            batch_file_name
        )

    return batch_demo_data, batch_goal_index, batch_valid_action_with_walk_index


def to_cuda_fn(data):
    batch_demo_data, batch_goal_index, batch_valid_action_with_walk_index = data

    if len(batch_demo_data)==4:
        batch_demo_index, batch_graph_length, batch_graph_input, batch_file_name = batch_demo_data

        batch_graph_input_class_objects = [[torch.tensor(j['class_objects']).cuda() for j in i] for i in batch_graph_input]
        batch_graph_input_object_coords = [[torch.tensor(j['object_coords']).cuda() for j in i] for i in batch_graph_input]
        batch_graph_input_states_objects = [[torch.tensor(j['states_objects']).cuda() for j in i] for i in batch_graph_input]
        batch_graph_input_mask_object = [[torch.tensor(j['mask_object']).cuda() for j in i] for i in batch_graph_input]
        
        batch_graph_input = { 'class_objects': batch_graph_input_class_objects, 
                                'object_coords': batch_graph_input_object_coords,
                                'states_objects': batch_graph_input_states_objects,
                                'mask_object': batch_graph_input_mask_object}

    else:
        batch_demo_index, batch_graph_length, batch_file_name = batch_demo_data
    

    batch_goal_index = [torch.tensor(i).cuda().long() for i in batch_goal_index]
    batch_valid_action_with_walk_index = [torch.tensor(i).cuda().long() for i in batch_valid_action_with_walk_index]
    
    if len(batch_demo_data)==4:
        batch_demo_data = (
            batch_demo_index,
            batch_graph_length, 
            batch_graph_input,
            batch_file_name
        )
    else:
        batch_demo_data = (
            batch_demo_index,
            batch_graph_length,
            batch_file_name
        )

    return batch_demo_data, batch_goal_index, batch_valid_action_with_walk_index





def one_hot(states, graph_node_states):
    one_hot = np.zeros(len(graph_node_states))
    for state in states:
        one_hot[graph_node_states[state]] = 1
    return one_hot


def build_graph(nodes, graph_class_names, max_node_length, graph_node_states):

    ## put character in the first node
    character_node = [node for node in nodes if 'character' in node['class_name']]
    other_nodes = [node for node in nodes if 'character' not in node['class_name']]
    assert len(other_nodes)+len(character_node) == len(nodes)

    if len(character_node)!=1:
        pdb.set_trace()
    nodes = character_node + other_nodes



    all_class_names = np.zeros((max_node_length)).astype(np.int32)
    class_names_str = [node['class_name'] for node in nodes]
    class_names = np.array([graph_class_names[class_name] for class_name in class_names_str])
    all_class_names[:len(nodes)] = class_names

    
    obj_coords = np.zeros((max_node_length, 6))
    char_coord = np.array(nodes[0]['bounding_box']['center'])
    rel_coords = [np.array([0,0,0])[None, :] if 'bounding_box' not in node.keys() else (np.array(node['bounding_box']['center']) - char_coord)[None, :] for node in nodes]
    bounds = [np.array([0,0,0])[None, :] if 'bounding_box' not in node.keys() else np.array(node['bounding_box']['size'])[None, :] for node in nodes]
    rel_coords = np.concatenate([rel_coords, bounds], axis=2)
    obj_coords[:len(nodes)] = np.concatenate(rel_coords, 0)


    all_node_states = np.zeros((max_node_length, len(graph_node_states)))
    node_states = np.array([one_hot(node['states'], graph_node_states) for node in nodes])
    all_node_states[:len(nodes)] = node_states

    mask_nodes = np.zeros((max_node_length))
    mask_nodes[:len(nodes)] = 1.


    output = {
        'class_objects': all_class_names,
        'object_coords': obj_coords,
        'states_objects': all_node_states,
        'mask_object': mask_nodes
    }
    
    return output



def gather_data(args):
    data_path = 'dataset/watch_data/gather_data_actiongraph_train.json'
    meta_data_path = 'dataset/watch_data/metadata.json'
    data_path_new_test = 'dataset/watch_data/gather_data_actiongraph_new_test.json'
    data_path_test = 'dataset/watch_data/gather_data_actiongraph_test.json'

    
    with open(data_path_new_test, 'r') as f:
        new_test_data = json.load(f)
        new_test_data = new_test_data['new_test_data']
    
    if os.path.exists(data_path):
        print('load gather_data, this may take a while...', data_path)
        with open(data_path_test, 'r') as f:
            data = json.load(f)
            test_data = data['test_data']

        if args.inference:
            train_data = data['test_data']
        else:
            with open(data_path, 'r') as f:
                data = json.load(f)
            train_data = data['train_data']


        with open(meta_data_path, 'r') as f:
            data = json.load(f)
            
            action_predicates = data['action_predicates']
            all_action = data['all_action']
            all_object = data['all_object']
            
            goal_objects = data['goal_objects']
            goal_targets = data['goal_targets']
            goal_predicates = data['goal_predicates']
            
            graph_class_names = data['graph_class_names']
            graph_node_states = data['graph_node_states']

            max_goal_length = data['max_goal_length']
            max_action_length = data['max_action_length']
            max_node_length = data['max_node_length']
            

    ## -----------------------------------------------------------------------------
    ## add action, goal, and graph node index
    ## -----------------------------------------------------------------------------
    max_subgoal_length = 1
    
    for traintest in [train_data, test_data, new_test_data]:
        for data in traintest:

            ## goal
            goal_index = []
            subgoal_dict = {}
            for subgoal in data['goal']:
                goal_index.append(goal_predicates[subgoal])

                if goal_predicates[subgoal] not in subgoal_dict:
                    subgoal_dict[goal_predicates[subgoal]] = 1
                else:
                    subgoal_dict[goal_predicates[subgoal]] += 1

            this_max_subgoal_length = np.max(list(subgoal_dict.values()))
            if this_max_subgoal_length>max_subgoal_length:
                max_subgoal_length = this_max_subgoal_length
            

            goal_index.sort()
            if len(goal_index) < max_goal_length:
                for i in range(max_goal_length-len(goal_index)):
                    goal_index.append(0)


            ## action
            assert len(data['graphs'])==len(data['valid_action_with_walk'])
            valid_action_with_walk_index = []
            for action in data['valid_action_with_walk']:
                action_name = action[0].split(' ')[0]
                object_name = action[0].split(' ')[1]
                predicate_name = ' '.join([action_name, object_name])
                valid_action_with_walk_index.append(action_predicates[predicate_name])


            if len(valid_action_with_walk_index) < max_action_length:
                for i in range(max_action_length-len(valid_action_with_walk_index)):
                    valid_action_with_walk_index.append(0)


            ## graph node
            graph_inputs = []
            for graph in data['graphs']:
                graph_input = build_graph(graph, graph_class_names, max_node_length, graph_node_states)
                graph_inputs.append(graph_input)


            graph_input = build_graph(data['graphs'][-1], graph_class_names, max_node_length, graph_node_states)
            if len(graph_inputs) < max_action_length:
                for i in range(max_action_length-len(graph_inputs)):
                    graph_inputs.append(graph_input)


            data['goal_index'] = np.array(goal_index)
            data['valid_action_with_walk_index'] = np.array(valid_action_with_walk_index)
            data['graph_input'] = graph_inputs
        
      


    for traintest in [train_data, test_data, new_test_data]:
        for data in traintest:
            ## multi classifier goal
            multi_classifier_goal_index = np.zeros([len(goal_predicates)])
            for subgoal in data['goal']:
                multi_classifier_goal_index[goal_predicates[subgoal]] += 1
            data['multi_classifier_goal_index'] = multi_classifier_goal_index


    print('--------------------------------------------------------------------------------')
    print('train_data', len(train_data))
    print('test_data', len(test_data))
    print('new_test_data', len(new_test_data))
    print('--------------------------------------------------------------------------------')

    return train_data, test_data, new_test_data, action_predicates, all_action, all_object, goal_objects, goal_targets, goal_predicates, graph_class_names, graph_node_states, max_goal_length, max_action_length, max_node_length, max_subgoal_length



class demo_dset(Dataset):

    def __init__(
            self,
            args,
            data, 
            action_predicates, all_action, all_object, goal_objects, goal_targets, goal_predicates, graph_class_names, graph_node_states, max_goal_length, max_action_length, max_node_length, max_subgoal_length):


        self.inputtype = args.inputtype
        self.multi_classifier = args.multi_classifier
        self.data = data
        
        self.action_predicates = action_predicates
        self.all_action = all_action
        self.all_object = all_object
        
        self.goal_objects = goal_objects
        self.goal_targets = goal_targets
        self.goal_predicates = goal_predicates
        self.num_goal_predicates = len(goal_predicates)

        self.max_goal_length = max_goal_length
        self.max_action_length = max_action_length
        self.max_subgoal_length = max_subgoal_length

        if self.inputtype=='graphinput':
            self.graph_class_names = graph_class_names
            self.graph_node_states = graph_node_states
            self.num_node_states = len(graph_node_states)
            self.max_node_length = max_node_length


        print('-----------------------------------------------------------------------------')
        print('num_goal_predicates', self.num_goal_predicates)
        print('max_goal_length', self.max_goal_length)
        print('max_action_length', max_action_length)
        
        if self.inputtype=='graphinput':
            print('num_node_states', self.num_node_states)
            print('max_node_length', max_node_length)
        print('-----------------------------------------------------------------------------')

        
    def __getitem__(self, index):
        data = self.data[index]
        data = self._preprocess_one_data(data)
        return data

    def __len__(self):
        return len(self.data)

    def _preprocess_one_data(self, data):

        if self.inputtype=='graphinput':
            init_graph = data['init_graph']
            graphs = data['graphs']
            graph_input = data['graph_input']

        if self.multi_classifier:
            goal_index = data['multi_classifier_goal_index']
        else:
            goal_index = data['goal_index']

        name = data['name']
        valid_action_with_walk_index = data['valid_action_with_walk_index']
        action_length = len(valid_action_with_walk_index)
        
        if self.inputtype=='graphinput':
            inputdata = (action_length, graph_input, name)
        else:
            inputdata = (action_length, name)

        data = [inputdata, goal_index, valid_action_with_walk_index]

        return data


        
