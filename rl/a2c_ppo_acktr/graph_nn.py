import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules as modules
from dgl import DGLGraph
import dgl
import dgl.function as fn
from functools import partial

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]

                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class GraphModel(nn.Module):
    def __init__(self, num_classes, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, max_nodes=100, num_hidden_layers=1):
        super(GraphModel, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.max_nodes = max_nodes
        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = None # self.create_features()
        self.feat_in = nn.Embedding(num_classes, h_dim)

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        #i2h = self.build_input_layer()
        i2h = self.build_hidden_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                activation=F.relu, is_input_layer=False)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, inputs):
        keys = ['class_objects', 'states_objects', 'edge_tuples', 'edge_classes', 'mask_object', 'mask_edge']
        [all_class_names, node_states,
         all_edge_ids, all_edge_types, 
         mask_nodes, mask_edges] = [torch.unbind(inputs[key]) for key in keys]
        num_envs = len(all_class_names)
        hs = []
        graphs = []
        for env_id in range(num_envs):
            g = DGLGraph()
            num_nodes = int(mask_nodes[env_id].sum().item())
            num_edges = int(mask_edges[env_id].sum().item())

            ids = all_class_names[env_id][:num_nodes]
            g.add_nodes(num_nodes)

            if num_edges > 0:
                edge_types = all_edge_types[env_id][:num_edges].long()
                try:
                    g.add_edges(all_edge_ids[env_id][:num_edges, 0].long(),
                                all_edge_ids[env_id][:num_edges, 1].long(),
                                {'rel_type': edge_types.long(),
                                    'norm': torch.ones((num_edges, 1)).to(edge_types.device)})
                except:
                    pdb.set_trace()
            if self.features is None:
                g.ndata['h'] = self.feat_in(ids.long())
            graphs.append(g)

        batch_graph = dgl.batch(graphs)

        for layer in self.layers:
            layer(batch_graph)
        graphs = dgl.unbatch(batch_graph)
        hs_list = []
        for graph in graphs:
            curr_graph = graph.ndata.pop('h').unsqueeze(0)
            curr_nodes = curr_graph.shape[1]
            curr_graph = F.pad(curr_graph, (0,0,0, self.max_nodes - curr_nodes), 'constant', 0.)
            hs_list.append(curr_graph)
        hs = torch.cat(hs_list, dim=0)
        return hs

class Transformer(nn.Module):
    def __init__(self, num_classes, num_nodes, in_feat, out_feat, dropout=0.1, activation='relu', nhead=1):
        super(Transformer, self).__init__()
        encoder_layer = nn.modules.TransformerEncoderLayer(d_model=in_feat, nhead=nhead,
                                                           dim_feedforward=out_feat, dropout=dropout,
                                                           activation=activation)
        self.transformer = nn.modules.TransformerEncoder(
            encoder_layer,
            num_layers=6,
            norm=nn.modules.normalization.LayerNorm(in_feat))

        self.class_embedding = nn.Embedding(num_classes, in_feat-3)
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs):
        keys = ['class_objects', 'states_objects', 'edge_tuples', 'edge_classes',
                'mask_object', 'mask_edge', 'object_coords']
        [all_class_names, node_states,
         all_edge_ids, all_edge_types,
         mask_nodes, mask_edges, coords] = [inputs[key] for key in keys]

        # inputs, combination of class names and coordinates
        inputs = self.class_embedding(all_class_names.long())
        inputs_and_coords = torch.cat((inputs, coords), dim=2)
        inputs_and_coords = inputs_and_coords.transpose(0,1)
        outputs = self.transformer(inputs_and_coords, src_key_padding_mask=mask_nodes.byte())
        outputs = outputs.squeeze(0).transpose(0,1)
        return outputs
