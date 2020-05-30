import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules as modules
from dgl import DGLGraph
import dgl
from torch.nn import init
import dgl.function as fn
from functools import partial

class GatedGraphConv(nn.Module):
    r"""Gated Graph Convolution layer from paper `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

    .. math::
        h_{i}^{0} & = [ x_i \| \mathbf{0} ]

        a_{i}^{t} & = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} & = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    n_steps : int
        Number of recurrent steps.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_steps,
                 n_etypes,
                 bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('relu')
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def forward(self, graph):
        """Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        #assert graph.is_homograph(), \
        #    "not a homograph; convert it with to_homo and pass in the edge type as argument"
        #graph = graph.local_var()
        # zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
        # feat = th.cat([feat, zero_pad], -1)

        for _ in range(self._n_steps):
            feat = graph.ndata['h']
            # graph.ndata['h'] = feat
            for i in range(self._n_etypes):
                eids = (graph.edata['rel_type'] == (i+1)).nonzero().view(-1)
                if len(eids) > 0:
                    graph.apply_edges(
                        lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                        eids
                    )
            graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
            a = graph.ndata.pop('a') # (N, D)
            feat = self.gru(a, feat)
            graph.ndata['h'] = feat
        return graph

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


class ClassAndStates(nn.Module):
    def __init__(self, num_classes, num_states, h_dim):
        super(ClassAndStates, self).__init__()
        self.class_encoding = nn.Embedding(num_classes, int(h_dim/2))
        self.state_embedding = nn.Linear(num_states, int(h_dim / 2))
        inp_dim = int(h_dim / 2)
        self.combine = nn.Sequential(nn.ReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.ReLU()
                                     )

    def forward(self, class_ids, states):
        class_nodes = self.class_encoding(class_ids)
        state_embedding = self.state_embedding(states)

        state_and_class = torch.cat([class_nodes, state_embedding], dim=1)
        output_embedding = self.combine(state_and_class)
        return output_embedding



class GraphModelGGNN(nn.Module):
    def __init__(self, num_classes, num_nodes, h_dim, out_dim, num_rels, num_states, k=3):
        super(GraphModelGGNN, self).__init__()
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_states = num_states
        self.k = k

        self.ggnn = GatedGraphConv(in_feats=h_dim,
                                   out_feats=out_dim,
                                   n_steps=k,
                                   n_etypes=num_rels)


        self.feat_in = ClassAndStates(num_classes, num_states, h_dim)
        self.class_encoding = self.feat_in.class_encoding

    def forward(self, inputs):
        keys = ['class_objects', 'states_objects', 'edge_tuples', 'edge_classes', 'mask_object', 'mask_edge']
        [all_class_names, node_states,
         all_edge_ids, all_edge_types,
         mask_nodes, mask_edges] = [torch.unbind(inputs[key]) for key in keys]
        num_envs = len(all_class_names)
        hs = []
        graphs = []
        #print('Graphs', num_envs)
        for env_id in range(num_envs):
            g = DGLGraph()
            num_nodes = int(mask_nodes[env_id].sum().item())
            num_edges = int(mask_edges[env_id].sum().item())
            #print(num_edges)
            ids = all_class_names[env_id][:num_nodes]
            node_states_curr = node_states[env_id][:num_nodes]
            g.add_nodes(num_nodes)

            if num_edges > 0:
                edge_types = all_edge_types[env_id][:num_edges].long()
                #try:
                g.add_edges(all_edge_ids[env_id][:num_edges, 0].long(),
                            all_edge_ids[env_id][:num_edges, 1].long(),
                            {'rel_type': edge_types.long()})
                            #     'norm': torch.ones((num_edges, 1)).to(edge_types.device)})
                #except:
                #    pdb.set_trace()
            feats_in = self.feat_in(ids.long(), node_states_curr)
            g.ndata['h'] = feats_in
            graphs.append(g)
        #print('----s')
        batch_graph = dgl.batch(graphs)
        #if len(graphs) > 1:
        #    pdb.set_trace()
        batch_graph = self.ggnn(batch_graph)
        graphs = dgl.unbatch(batch_graph)
        hs_list = []
        # pdb.set_trace()
        for graph in graphs:
            curr_graph = graph.ndata.pop('h').unsqueeze(0)
            curr_nodes = curr_graph.shape[1]
            curr_graph = F.pad(curr_graph, (0,0,0, self.num_nodes - curr_nodes), 'constant', 0.)
            hs_list.append(curr_graph)
        hs = torch.cat(hs_list, dim=0)
        return hs


class GraphModel(nn.Module):
    def __init__(self, num_classes, num_nodes, h_dim, out_dim, num_rels, num_states,
                 num_bases=-1, num_hidden_layers=1):
        super(GraphModel, self).__init__()
        self.num_states = num_states
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes = num_nodes
        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = None # self.create_features()

        self.feat_in = ClassAndStates(num_classes, num_states, h_dim)
        #self.feat_in = nn.Embedding(num_classes, h_dim)

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
        #h2o = self.build_output_layer()
        #self.layers.append(h2o)

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
            node_states_curr = node_states[env_id][:num_nodes]
            g.add_nodes(num_nodes)

            if num_edges > 0:

                edge_types = all_edge_types[env_id][:num_edges].long()
                # try:
                g.add_edges(all_edge_ids[env_id][:num_edges, 0].long(),
                            all_edge_ids[env_id][:num_edges, 1].long(),
                            {'rel_type': edge_types.long(),
                                'norm': torch.ones((num_edges, 1)).to(edge_types.device)})
                # except:
                #     pdb.set_trace()
            if self.features is None:
                feats_in = self.feat_in(ids.long(), node_states_curr)
                g.ndata['h'] = feats_in
            graphs.append(g)

        batch_graph = dgl.batch(graphs)

        for layer in self.layers:
            layer(batch_graph)
        graphs = dgl.unbatch(batch_graph)
        hs_list = []
        for graph in graphs:
            curr_graph = graph.ndata.pop('h').unsqueeze(0)
            curr_nodes = curr_graph.shape[1]
            curr_graph = F.pad(curr_graph, (0,0,0, self.num_nodes - curr_nodes), 'constant', 0.)
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
        self.bad_transformer = False
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, mask_nodes):
        if not self.bad_transformer:
            mask_nodes = 1 - mask_nodes
        outputs = self.transformer(inputs.transpose(0,1), src_key_padding_mask=mask_nodes.bool())
        outputs = outputs.squeeze(0).transpose(0,1)
        return outputs



