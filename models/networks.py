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

            node_embedding = torch.zeros(node_embedding.shape)

            node_embedding = node_embedding.scatter_add(1, final_node[:, :, None].repeat(1, 1, new_embed.shape[-1]), new_embed)
        return node_embedding

class ClassNameStateRepresentation(nn.Module):
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
        is_cuda = next(self.parameters()).is_cuda
        node_names = [node['class_name'] for node in observations['nodes']]
        # ids in the embedding
        node_name_ids = torch.tensor(
            [self.dataset.object_dict.get_id(node_name) for node_name in node_names]).long()
        state_one_hot = np.zeros((len(node_names), self.num_states))
        for node_it, node in enumerate(observations['nodes']):
            state_ids = [self.dataset.state_dict.get_id(state) for state in node['states']]
            for state_id in state_ids: state_one_hot[node_it, state_id] = 1
        state_one_hot = torch.tensor(state_one_hot).float()
        state_count = state_one_hot.sum(1)+self.eps
        state_embedding = self.state_embedding(state_one_hot)/state_count[:, None]
        return state_embedding, None


class StateRepresentation(nn.Module):
    def __init__(self, helper, dataset):
        super().__init__()
        self.dataset = dataset
        self.helper = helper
        self.object_embedding = nn.Embedding(len(dataset.object_dict), helper.args.object_dim)

    def forward(self, observations):
        is_cuda = next(self.parameters()).is_cuda
        node_names = [node['class_name']  for node in observations['nodes']]
        # ids in the embedding
        node_name_ids = torch.tensor(
            [self.dataset.object_dict.get_id(node_name) for node_name in node_names]).long()

        char_id = torch.tensor([self.dataset.object_dict.get_id('character')]).long()
        if is_cuda:
            node_name_ids = node_name_ids.cuda()
            char_id = char_id.cuda()

        node_embeddings = self.object_embedding(node_name_ids)
        char_embedding = self.object_embedding(char_id)
        return node_embeddings, char_embedding






class GraphStateRepresentation(nn.Module):
    def __init__(self, helper, dataset):
        super().__init__()
        self.dataset = dataset
        self.helper = helper
        self.vector_repr = ClassNameStateRepresentation(self.helper, self.dataset)
        self.graph_encoding = GCNN(2, helper.args.relation_dim, len(dataset.relation_dict))

    def forward(self, observations):
        is_cuda = next(self.parameters()).is_cuda

        # Convert observations into a tensor. This should probably go out of this function...
        # Node ids to model id
        node_ids = [node['id'] for node in observations['nodes']]
        node_names = [node['class_name'] for node in observations['nodes']]
        idgraph2idmodel = utils.DictObjId(node_ids, include_other=False)
        num_nodes = len(observations['nodes'])
        num_edges = len(observations['edges'])
        nodes_connected = np.zeros((num_edges, 2))
        edges_connecting = np.zeros((num_edges))
        for it, edge in enumerate(observations['edges']):
            nodes_connected[it, 0] = idgraph2idmodel.get_id(edge['from_id'])
            nodes_connected[it, 1] = idgraph2idmodel.get_id(edge['to_id'])
            edges_connecting[it] = self.dataset.relation_dict.get_id(edge['relation_type'])

        nodes_connected = torch.tensor(nodes_connected)[None, :] # We do it batched here
        edge_types = torch.tensor(edges_connecting)[None, :] # We do it batched here

        # node representation
        initial_node_repr = self.vector_repr(observations)[0]
        node_embeddings = self.graph_encoding(initial_node_repr[None, :], nodes_connected, edge_types)
        node_embeddings = node_embeddings[0] # We do single batches...
        char_it = [it for it,x in enumerate(observations['nodes']) if x['class_name'] == 'character'][0]
        state_embedding = node_embeddings[char_it]
        pdb.set_trace()
        return node_embeddings, state_embedding