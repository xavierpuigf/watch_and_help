import random
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from helper import fc_block, Constant
import ipdb
import pdb
import torchvision



def _calculate_accuracy_predicate(logits, batch_target, max_possible_count=None, topk=1, multi_classifier=False):
    batch_size = batch_target.size(0) / max_possible_count

    _, pred = logits.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(batch_target.view(1, -1).expand_as(pred))

    k = 1    
    accuray = correct[:k].view(-1).float()
    accuray = accuray.view(-1, max_possible_count)

    correct_k = (accuray.sum(1)==max_possible_count).sum(0)
    correct_k = correct_k * (100.0 / batch_size)
    
    return correct_k

def _calculate_accuracy(
        action_correct,
        object_correct,
        rel_correct,
        target_correct,
        batch_length,
        info):
    
    action_valid_correct = [sum(action_correct[i, :(l - 1)])
                            for i, l in enumerate(batch_length)]
    object_valid_correct = [sum(object_correct[i, :(l - 1)])
                             for i, l in enumerate(batch_length)]
    rel_valid_correct = [sum(rel_correct[i, :(l - 1)])
                             for i, l in enumerate(batch_length)]
    target_valid_correct = [sum(target_correct[i, :(l - 1)])
                             for i, l in enumerate(batch_length)]

    action_accuracy = sum(action_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))
    object_accuracy = sum(object_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))
    rel_accuracy = sum(rel_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))
    target_accuracy = sum(target_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))

    info.update({'action_accuracy': action_accuracy.cpu().item()})
    info.update({'object_accuracy': object_accuracy.cpu().item()})
    info.update({'rel_accuracy': rel_accuracy.cpu().item()})
    info.update({'target_accuracy': target_accuracy.cpu().item()})




class PredicateClassifier(nn.Module):

    def __init__(
            self, 
            args,
            dset):

        super(PredicateClassifier, self).__init__()

        self.num_goal_predicates = dset.num_goal_predicates
        self.max_possible_count = dset.max_goal_length

        hidden_size = args.demo_hidden
        print('hidden_size', hidden_size)

        if args.dropout==0:
            print('dropout', args.dropout)
            classifier = nn.Sequential()
            classifier.add_module('fc_block1', fc_block(hidden_size, hidden_size, False, nn.Tanh))
            classifier.add_module('fc_block2', fc_block(hidden_size, self.max_possible_count*self.num_goal_predicates, False, None))
        else:
            print('dropout not 0', args.dropout)
            classifier = nn.Sequential()
            classifier.add_module('fc_block1', fc_block(hidden_size, hidden_size, False, nn.Tanh))
            classifier.add_module('dropout', nn.Dropout(args.dropout))
            classifier.add_module('fc_block2', fc_block(hidden_size, self.max_possible_count*self.num_goal_predicates, False, None))

        self.classifier = classifier

    def forward(self, bs, input_emb, batch_target, batch_file_name, **kwargs):
        
        logits = self.classifier(input_emb)
        logits = logits.reshape([-1, self.num_goal_predicates])
        prob = F.softmax(logits, 1)

        batch_target = torch.cat(batch_target)
        loss = F.cross_entropy(logits, batch_target)  

        top1 = _calculate_accuracy_predicate(logits, batch_target, self.max_possible_count)

        with torch.no_grad():
            info = {
                "prob": prob.cpu().numpy(),
                "loss": loss.cpu().numpy(),
                "top1": top1.cpu().numpy(),
                "target": batch_target.cpu().numpy(),
                "file_name": batch_file_name
            }

        return loss, info



class PredicateClassifierMultiClassifier(nn.Module):

    def __init__(
            self, 
            args,
            dset):

        super(PredicateClassifierMultiClassifier, self).__init__()

        self.num_goal_predicates = dset.num_goal_predicates
        self.max_possible_count = dset.max_goal_length
        self.max_subgoal_length = dset.max_subgoal_length

        hidden_size = args.demo_hidden
        print('hidden_size', hidden_size)
        print('PredicateClassifierMultiClassifier')

        if args.dropout==0:
            print('dropout', args.dropout)
            classifier = nn.Sequential()
            classifier.add_module('fc_block1', fc_block(hidden_size, hidden_size, False, nn.Tanh))
            classifier.add_module('fc_block2', fc_block(hidden_size, self.num_goal_predicates*(self.max_subgoal_length+1), False, None))
        else:
            print('dropout not 0', args.dropout)
            classifier = nn.Sequential()
            classifier.add_module('fc_block1', fc_block(hidden_size, hidden_size, False, nn.Tanh))
            classifier.add_module('dropout', nn.Dropout(args.dropout))
            classifier.add_module('fc_block2', fc_block(hidden_size, self.num_goal_predicates*(self.max_subgoal_length+1), False, None))

        
        self.classifier = classifier

    def forward(self, bs, input_emb, batch_target, batch_file_name, **kwargs):
        
        logits = self.classifier(input_emb)
        logits = logits.reshape([-1, (self.max_subgoal_length+1)])
        prob = F.softmax(logits, 1)

        batch_target = torch.cat(batch_target)
        loss = F.cross_entropy(logits, batch_target)  

        top1 = _calculate_accuracy_predicate(logits, batch_target, self.num_goal_predicates, multi_classifier=True)

        with torch.no_grad():
            info = {
                "prob": prob.cpu().numpy(),
                "loss": loss.cpu().numpy(),
                "top1": top1.cpu().numpy(),
                "target": batch_target.cpu().numpy(),
                "file_name": batch_file_name
            }

        return loss, info




class ActionDemoEncoder(nn.Module):

    def __init__(self, args, dset, pooling):

        super(ActionDemoEncoder, self).__init__()

        # Image encoder
        import torchvision

        hidden_size = args.demo_hidden

        len_action_predicates = len(dset.action_predicates)
        self.action_embed = nn.Embedding(len_action_predicates, hidden_size)

        feat2hidden = nn.Sequential()
        feat2hidden.add_module(
            'fc_block1', fc_block(hidden_size, hidden_size, False, nn.ReLU))
        self.feat2hidden = feat2hidden

        self.pooling = pooling

        if 'lstm' in self.pooling:
            self.lstm = nn.LSTM(hidden_size, hidden_size)


    def forward(self, batch_length, batch_file_name, batch_valid_action_with_walk_index):
        
        stacked_demo = torch.cat(batch_valid_action_with_walk_index, 0)
        stacked_demo_feat = self.action_embed(stacked_demo)
        
        stacked_demo_feat = self.feat2hidden(stacked_demo_feat)

        
        batch_demo_feat = []
        start = 0
        assert np.sum(batch_length) == stacked_demo_feat.shape[0]
        for length in batch_length:
            feat = stacked_demo_feat[start:(start + length), :]
            if len(feat.size()) == 3:
                feat = feat.unsqueeze(0)
            start += length
            
            if self.pooling == 'max':
                feat = torch.max(feat, 0)[0]
            elif self.pooling == 'avg':
                feat = torch.mean(feat, 0)
            elif self.pooling == 'lstmavg':
                lstm_out, hidden = self.lstm(feat.view(len(feat), 1, -1))
                lstm_out = lstm_out.view(len(feat), -1)
                feat = torch.mean(lstm_out, 0)
            elif self.pooling == 'lstmlast':
                lstm_out, hidden = self.lstm(feat.view(len(feat), 1, -1))
                lstm_out = lstm_out.view(len(feat), -1)
                feat = lstm_out[-1]
            else:
                raise ValueError


            batch_demo_feat.append(feat)

        demo_emb = torch.stack(batch_demo_feat, 0)
        
        return demo_emb, batch_demo_feat




            

class Transformer(nn.Module):
    def __init__(self, num_classes, num_nodes, in_feat, out_feat, dropout=0.2, activation='relu', nhead=2):
        super(Transformer, self).__init__()
        
        print('head', nhead)

        encoder_layer = nn.modules.TransformerEncoderLayer(d_model=in_feat, nhead=nhead, dropout=dropout, dim_feedforward=out_feat)
        self.transformer = nn.modules.TransformerEncoder(
            encoder_layer,
            num_layers=1)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, mask_nodes):

        mask_nodes = 1 - mask_nodes
        outputs = self.transformer(inputs.transpose(0,1), src_key_padding_mask=mask_nodes.bool())

        outputs = outputs.squeeze(0).transpose(0,1)
        return outputs




class ObjNameCoordStateEncode(nn.Module):
    def __init__(self, output_dim=128, num_classes=50, num_states=4):
        super(ObjNameCoordStateEncode, self).__init__()


        assert output_dim % 2 == 0
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Sequential(nn.Embedding(num_classes, int(output_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(output_dim / 2), int(output_dim / 2)))


        self.state_embedding = nn.Sequential(nn.Linear(num_states, int(output_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(output_dim / 2), int(output_dim / 2)))


        self.coord_embedding = nn.Sequential(nn.Linear(6, int(output_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(output_dim / 2), int(output_dim / 2)))


        inp_dim = int(output_dim + output_dim/2)
        

        self.combine = nn.Sequential(nn.ReLU(),
                                     nn.Linear(inp_dim, output_dim))
        

    def forward(self, class_ids, coords, state):
        
        state_embedding = self.state_embedding(state)
        class_embedding = self.class_embedding(class_ids)
        coord_embedding = self.coord_embedding(coords)

        inp = torch.cat([class_embedding, coord_embedding, state_embedding], dim=2)

        return self.combine(inp)



class GraphDemoEncoder(nn.Module):

    def __init__(self, args, dset, pooling):

        super(GraphDemoEncoder, self).__init__()

        # Image encoder
        hidden_size = args.demo_hidden
        max_nodes = dset.max_node_length
        num_classes = len(dset.graph_class_names)
        num_states = len(dset.graph_node_states)

        self.main = Transformer(nhead=args.transformer_nhead, num_classes=num_classes, num_nodes=max_nodes, in_feat=hidden_size, out_feat=hidden_size)
        self.single_object_encoding = ObjNameCoordStateEncode(output_dim=hidden_size, num_classes=num_classes, num_states=num_states)
        self.object_class_encoding = self.single_object_encoding.class_embedding
        self.pooling = pooling

        if 'lstm' in self.pooling:
            self.lstm = nn.LSTM(hidden_size, hidden_size)
            self.lstm_reverse = nn.LSTM(hidden_size, hidden_size)
            
    def forward(self, batch_length, batch_graph_input, batch_file_name, batch_valid_action_with_walk_index):

        class_objects = batch_graph_input['class_objects']
        object_coords = batch_graph_input['object_coords']
        states_objects = batch_graph_input['states_objects']
        mask_object = batch_graph_input['mask_object']

        
        stacked_demo_class_objects = [torch.stack(demo, 0) for demo in class_objects]
        stacked_demo_class_objects = torch.cat(stacked_demo_class_objects, 0)

        stacked_demo_object_coords = [torch.stack(demo, 0) for demo in object_coords]
        stacked_demo_object_coords = torch.cat(stacked_demo_object_coords, 0)

        stacked_demo_states_objects = [torch.stack(demo, 0) for demo in states_objects]
        stacked_demo_states_objects = torch.cat(stacked_demo_states_objects, 0)

        stacked_demo_mask_object = [torch.stack(demo, 0) for demo in mask_object]
        stacked_demo_mask_object = torch.cat(stacked_demo_mask_object, 0)


        input_node_embedding = self.single_object_encoding(stacked_demo_class_objects.long(),
                                                           stacked_demo_object_coords.float(),
                                                           stacked_demo_states_objects.float()).squeeze(1)
        
        stacked_demo_feat = self.main(input_node_embedding, stacked_demo_mask_object)

        ## average over nodes
        stacked_demo_feat = torch.mean(stacked_demo_feat, 1)


        ## average/max over temporal
        assert np.sum(batch_length) == stacked_demo_feat.shape[0]
        batch_demo_feat = []
        start = 0
        for length in batch_length:
            feat = stacked_demo_feat[start:(start + length), :]

            
            start += length
            if self.pooling == 'max':
                feat = torch.max(feat, 0)[0]
            elif self.pooling == 'avg':
                feat = torch.mean(feat, 0)
            elif self.pooling == 'lstmavg':
                lstm_out, _ = self.lstm(feat.view(len(feat), 1, -1))
                lstm_out = lstm_out.view(len(feat), -1)
                feat = torch.mean(lstm_out, 0)
            else:
                raise ValueError

            batch_demo_feat.append(feat)

        
        demo_emb = torch.stack(batch_demo_feat, 0)
        return demo_emb, batch_demo_feat



