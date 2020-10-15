import copy
import numpy as np
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import pdb


def _align_tensor_index(reference_index, tensor_index):

    where_in_tensor = []
    for i in reference_index:
        where = np.where(i == tensor_index)[0][0]
        where_in_tensor.append(where)
    return np.array(where_in_tensor)


def _sort_by_length(list_of_tensor, batch_length, return_idx=False):
    
    idx = np.argsort(np.array(copy.copy(batch_length)))[::-1]
    for i, tensor in enumerate(list_of_tensor):
        if isinstance(tensor, dict):
            list_of_tensor[i]['class_objects'] = [tensor['class_objects'][j] for j in idx]
            list_of_tensor[i]['object_coords'] = [tensor['object_coords'][j] for j in idx]
            list_of_tensor[i]['states_objects'] = [tensor['states_objects'][j] for j in idx]
            list_of_tensor[i]['mask_object'] = [tensor['mask_object'][j] for j in idx]
        else:
            list_of_tensor[i] = [tensor[j] for j in idx]
    if return_idx:
        return list_of_tensor, idx
    else:
        return list_of_tensor


def _sort_by_index(list_of_tensor, idx):
    for i, tensor in enumerate(list_of_tensor):
        list_of_tensor[i] = [tensor[j] for j in idx]
    return list_of_tensor




class GraphDemo2Predicate(nn.Module):

    summary_keys = ['loss', 'top1']
    def __init__(self, args, dset, **kwargs):
        from network.module_graph import PredicateClassifier, PredicateClassifierMultiClassifier
        super(GraphDemo2Predicate, self).__init__()

        print('------------------------------------------------------------------------------------------')
        print('GraphDemo2Predicate')
        print('------------------------------------------------------------------------------------------')


        self.inputtype = args.inputtype
        self.multi_classifier = args.multi_classifier
        model_type = kwargs["model_type"]
        print('model_type', model_type)

        if model_type.lower() == 'max':
            from network.module_graph import GraphDemoEncoder
            demo_encoder = GraphDemoEncoder(args, dset, 'max')
        elif model_type.lower() == 'avg':
            from network.module_graph import GraphDemoEncoder
            demo_encoder = GraphDemoEncoder(args, dset, 'avg')
        elif model_type.lower() == 'lstmavg':
            from network.module_graph import GraphDemoEncoder
            demo_encoder = GraphDemoEncoder(args, dset, 'lstmavg')
        elif model_type.lower() == 'bilstmavg':
            from network.module_graph import GraphDemoEncoder
            demo_encoder = GraphDemoEncoder(args, dset, 'bilstmavg')
        elif model_type.lower() == 'lstmlast':
            from network.module_graph import GraphDemoEncoder
            demo_encoder = GraphDemoEncoder(args, dset, 'lstmlast')
        elif model_type.lower() == 'bilstmlast':
            from network.module_graph import GraphDemoEncoder
            demo_encoder = GraphDemoEncoder(args, dset, 'bilstmlast')
        else:
            raise ValueError
        demo_encoder = torch.nn.DataParallel(demo_encoder)

        if self.multi_classifier:
            predicate_decoder = PredicateClassifierMultiClassifier(args, dset)
        else:
            predicate_decoder = PredicateClassifier(args, dset)
        
        # for quick save and load
        all_modules = nn.Sequential()
        all_modules.add_module('demo_encoder', demo_encoder)
        all_modules.add_module('predicate_decoder', predicate_decoder)

        self.demo_encoder = demo_encoder
        self.predicate_decoder = predicate_decoder
        self.all_modules = all_modules
        self.to_cuda_fn = None


    def set_to_cuda_fn(self, to_cuda_fn):
        self.to_cuda_fn = to_cuda_fn

    def forward(self, data, **kwargs):
        '''
            Note: The order of the `data` won't change in this function
        '''
        
        if self.to_cuda_fn:
            data = self.to_cuda_fn(data)
        
        # demonstration
        # sort according to the length of demonstration
        batch_graph_length = data[0][1]
        batch_goal_index = data[1]
        batch_valid_action_with_walk_index = data[2]

        batch_demo = _sort_by_length(list(data[0]), batch_graph_length)
        batch_goal_index = _sort_by_length([batch_goal_index], batch_graph_length)[0]
        batch_valid_action_with_walk_index = _sort_by_length([batch_valid_action_with_walk_index], batch_graph_length)[0]

        batch_demo_index, batch_graph_length, batch_graph_input, batch_file_name = batch_demo
        batch_demo_emb, _ = self.demo_encoder(batch_graph_length, batch_graph_input, batch_file_name, batch_valid_action_with_walk_index)

        bs = len(batch_graph_length)
        
        loss, info = self.predicate_decoder(bs, batch_demo_emb, batch_goal_index, batch_file_name)

        return loss, info
                    
    def write_summary(self, writer, info, postfix):

        model_name = 'Demo2Predicate-{}/'.format(postfix)
        for k in self.summary_keys:
            if k in info.keys():
                writer.scalar_summary(model_name + k, info[k])

    def save(self, path, verbose=False):
        
        if verbose:
            print(colored('[*] Save model at {}'.format(path), 'magenta'))
        torch.save(self.all_modules.state_dict(), path)

    def load(self, path, verbose=False):

        if verbose:
            print(colored('[*] Load model at {}'.format(path), 'magenta'))
        self.all_modules.load_state_dict(
            torch.load(
                path,
                map_location=lambda storage,
                loc: storage))





class ActionDemo2Predicate(nn.Module):

    summary_keys = ['loss', 'top1']
    def __init__(self, args, dset, **kwargs):
        from network.module_graph import PredicateClassifier
        super(ActionDemo2Predicate, self).__init__()

        print('------------------------------------------------------------------------------------------')
        print('ActionDemo2Predicate')
        print('------------------------------------------------------------------------------------------')

        model_type = kwargs["model_type"]
        print('model_type', model_type)

        if model_type.lower() == 'max':
            from network.module_graph import ActionDemoEncoder
            demo_encoder = ActionDemoEncoder(args, dset, 'max')
        elif model_type.lower() == 'avg':
            from network.module_graph import ActionDemoEncoder
            demo_encoder = ActionDemoEncoder(args, dset, 'avg')
        elif model_type.lower() == 'lstmavg':
            from network.module_graph import ActionDemoEncoder
            demo_encoder = ActionDemoEncoder(args, dset, 'lstmavg')
        elif model_type.lower() == 'bilstmavg':
            from network.module_graph import ActionDemoEncoder
            demo_encoder = ActionDemoEncoder(args, dset, 'bilstmavg')
        elif model_type.lower() == 'lstmlast':
            from network.module_graph import ActionDemoEncoder
            demo_encoder = ActionDemoEncoder(args, dset, 'lstmlast')
        elif model_type.lower() == 'bilstmlast':
            from network.module_graph import ActionDemoEncoder
            demo_encoder = ActionDemoEncoder(args, dset, 'bilstmlast')
        else:
            raise ValueError
        demo_encoder = torch.nn.DataParallel(demo_encoder)

        predicate_decoder = PredicateClassifier(args, dset)
        
        # for quick save and load
        all_modules = nn.Sequential()
        all_modules.add_module('demo_encoder', demo_encoder)
        all_modules.add_module('predicate_decoder', predicate_decoder)

        self.demo_encoder = demo_encoder
        self.predicate_decoder = predicate_decoder
        self.all_modules = all_modules
        self.to_cuda_fn = None

    def set_to_cuda_fn(self, to_cuda_fn):
        self.to_cuda_fn = to_cuda_fn

    def forward(self, data, **kwargs):
        '''
            Note: The order of the `data` won't change in this function
        '''
        
        if self.to_cuda_fn:
            data = self.to_cuda_fn(data)
        
        # demonstration
        # sort according to the length of demonstration
        batch_graph_length = data[0][1]
        batch_goal_index = data[1]
        batch_valid_action_with_walk_index = data[2]

        batch_demo = _sort_by_length(list(data[0]), batch_graph_length)
        batch_goal_index = _sort_by_length([batch_goal_index], batch_graph_length)[0]
        batch_valid_action_with_walk_index = _sort_by_length([batch_valid_action_with_walk_index], batch_graph_length)[0]

        # demonstration encoder
        batch_demo_index, batch_graph_length, batch_file_name = batch_demo
        batch_demo_emb, _ = self.demo_encoder(batch_graph_length, batch_file_name, batch_valid_action_with_walk_index)

        # predicate decoder
        bs = len(batch_graph_length)

        loss, info = self.predicate_decoder(bs, batch_demo_emb, batch_goal_index, batch_file_name)

        return loss, info
                    
    def write_summary(self, writer, info, postfix):

        model_name = 'Demo2Predicate-{}/'.format(postfix)
        for k in self.summary_keys:
            if k in info.keys():
                writer.scalar_summary(model_name + k, info[k])

    def save(self, path, verbose=False):
        
        if verbose:
            print(colored('[*] Save model at {}'.format(path), 'magenta'))
        torch.save(self.all_modules.state_dict(), path)

    def load(self, path, verbose=False):

        if verbose:
            print(colored('[*] Load model at {}'.format(path), 'magenta'))
        self.all_modules.load_state_dict(
            torch.load(
                path,
                map_location=lambda storage,
                loc: storage))




