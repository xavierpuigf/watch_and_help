from torch import nn
import torch
import utils
import numpy as np
import pdb


class GCNN(nn.Module):
    def __init__(self, num_layers, edge_dim, num_edge_types):
        super(GCNN, self).__init__()
        self.num_layers = num_layers
        self.edge_embedding = nn.Embedding(num_edge_types, edge_dim)
        self.combi_layers = []
        self.sigmoid = nn.Sigmoid()
        for _ in range(num_layers):
            combi_layer = nn.Linear(edge_dim*2, edge_dim)
            self.combi_layers.append(combi_layer)

    def forward(self, node_embedding, edges, edge_types):
        # TODO: there should be some edge for rooms
        '''

        :param node_embedding: [batch, num_node, dim_embedding]
        :param edges: [batch, num_edges, 2]: id_node, id_node]
        :param edge_types: [bathc, num_edges, 1]: type_node
        :return:
        '''
        bs, num_nodes = node_embedding.shape[:2]
        num_edges = edges.shape[1]
        edge_graph_embed = self.edge_embedding(edge_types.long())
        origin_node = edges[:, :, 0].long()
        final_node = edges[:, :, 1].long()
        for i in range(self.num_layers):
            # Embeddig for every source node + edge

            # batch indexing
            node_embed_flat = node_embedding.view(-1, node_embedding.shape[-1])

            origin_node_flat = (origin_node + torch.arange(bs)[:, None]*num_nodes).view(-1)
            node_and_edge = torch.cat([node_embed_flat[origin_node_flat].view(bs, num_edges, -1),
                                       edge_graph_embed], dim=2)
            new_embed = self.combi_layers[i](node_and_edge.view(bs*num_edges, -1)).view(bs, num_edges, -1)

            node_embedding_res = torch.zeros(node_embedding.shape).type_as(node_embedding)

            node_embedding_res = node_embedding_res.scatter_add(1, final_node[:, :, None].repeat(1, 1, new_embed.shape[-1]), new_embed)
            node_embedding_res = self.sigmoid(node_embedding_res)
            node_embedding += node_embedding_res
        return node_embedding





class ClassNameStateRepresentation(nn.Module):
    # Representaiton based on class names and state
    def __init__(self, helper, dataset):
        super().__init__()
        self.dataset = dataset
        self.helper = helper
        self.eps = 1e-6
        self.num_states = len(dataset.state_dict)
        self.object_embedding = nn.Embedding(len(dataset.object_dict), helper.args.object_dim)
        self.state_embedding = nn.Linear(self.num_states, helper.args.state_dim)
        self.combine_state_dim = nn.Sequential(torch.nn.Linear(helper.args.state_dim+helper.args.object_dim,
                                                               helper.args.object_dim),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(helper.args.object_dim, helper.args.object_dim))
    def forward(self, observations):
        class_names, states, edges, edge_types, visibility, mask_edges = observations

        node_name_embedding = self.object_embedding(class_names)
        state_embedding = self.state_embedding(states.float())
        state_and_node = torch.cat([node_name_embedding, state_embedding], -1)
        representation = self.combine_state_dim(state_and_node)
        return representation



##### State Representations ####




class GraphStateRepresentation(nn.Module):
    def __init__(self, helper, dataset):
        super().__init__()
        self.dataset = dataset
        self.helper = helper
        self.initial_node_repr = ClassNameStateRepresentation(self.helper, self.dataset)
        self.in_fts = 100
        self.out_fts = 100
        self.graph_encoding = GatedGraphConv(self.in_fts, self.out_fts, 3, len(dataset.relation_dict))# GCNN(2, helper.args.relation_dim, len(dataset.relation_dict))
        self.gru = nn.GRUCell(self.out_fts, self.out_fts, bias=False)

    def forward(self, observations):
        class_names, states, edge_values, edge_types, visibility, mask_edges = observations
        # Obtain the initial node representation [bs, timesteps, num_nodes, node_dim]
        initial_node_repr = self.initial_node_repr(observations)

        # separate timesteps
        init_nodes = torch.unbind(initial_node_repr, 1)
        edges_tsp = torch.unbind(edge_values, 1)
        edge_types = torch.unbind(edge_types, 1)
        mask_edges = torch.unbind(mask_edges,1)
        visibility_tsp = torch.unbind(visibility, 1)
        bs = class_names.shape[0]

        node_representations = []
        # goal_representation TODO: ser here
        h_0_node = torch.zeros(init_nodes[0].shape)

        if initial_node_repr.is_cuda:
            h_0_node = h_0_node.cuda()

        h_t = h_0_node

        h_t_flat = h_t.view(-1, h_t.shape[-1])
        for it in range(len(init_nodes)):
            node_representation_step_flat = self.gru(init_nodes[it].view(-1, self.out_fts), h_t_flat)
            h_t_flat = node_representation_step_flat
            node_representation_step = node_representation_step_flat.view(
                bs, -1, node_representation_step_flat.shape[-1])
            node_repr = self.graph_encoding(node_representation_step, edges_tsp[it], edge_types[it], mask_edges[it])

            # Mask out the nodes by their visibility
            node_repr = node_repr * visibility_tsp[it][:, :, None]
            node_representations.append(node_repr)


        node_representations = torch.cat([x.unsqueeze(1) for x in node_representations], 1)
        num_nodes = visibility.sum(-1) + 1e-9
        global_repr = (node_representations * visibility.unsqueeze(-1)).sum(2)/num_nodes[:, :, None]

        return node_representations, global_repr



## GNN ##


class GatedGraphConv(nn.Module):
    # Code from https://github.com/dmlc/dgl/blob/ddb5d804f94ccdbe12d329454d7bc413ee7b181b/python/dgl/nn/pytorch/conv/gatedgraphconv.py
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
        self.edge_embed = nn.Embedding(n_etypes, out_feats * out_feats)
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        #gain = init.calculate_gain('relu')
        self.gru.reset_parameters()
        #init.xavier_normal_(self.edge_embed.weight, gain=gain)

    def forward(self, feat, edges, edge_types, mask_edges):
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
        # graph = graph.local_var()
        # pdb.set_trace()
        zero_pad = feat.new_zeros((feat.shape[0], feat.shape[1], self._out_feats - feat.shape[2]))
        feat = torch.cat([feat, zero_pad], -1)

        bs, num_nodes = feat.shape[:2]

        # Flatten the batch
        origin_nodes = edges[:, :, 0]
        dest_nodes = edges[:, :, 1]
        range_bs = torch.arange(bs)[:, None]
        if feat.is_cuda:
            range_bs = range_bs.cuda()

        origin_nodes_flat = (origin_nodes + (num_nodes * range_bs)).view(-1)
        dest_nodes_flat = (dest_nodes + (num_nodes * range_bs)).view(-1)
        feat_flat = feat.view(-1, feat.shape[-1])

        for _ in range(self._n_steps):

            edge_embeddings = self.edge_embed(edge_types).view(-1, self._out_feats, self._out_feats)

            # Mask out edge_embeddings
            edge_embeddings *= mask_edges.view(-1)[:, None, None]

            feat_input_embeddings = feat_flat[origin_nodes_flat][:, None, :]
            input_embeddings = (edge_embeddings * feat_input_embeddings).sum(-1)

            a_t = torch.zeros((num_nodes*bs, self._out_feats))
            if torch.cuda.is_available():
                a_t = a_t.cuda()
            a_t = a_t.scatter_add(0, dest_nodes_flat[:, None].repeat(1, self._out_feats), input_embeddings)

            feat = self.gru(a_t,  feat_flat)
        feat = feat.view(bs, num_nodes, -1)
        return feat