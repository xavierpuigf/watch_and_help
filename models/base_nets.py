from torch import nn
from .graph_nn import Transformer, GraphModel, GraphModelGGNN
import pdb
from utils.utils_models import init
import torch

# class NNBase(nn.Module):
#     def __init__(self, recurrent, recurrent_input_size, hidden_size):
#         super(NNBase, self).__init__()

#         self._hidden_size = hidden_size
#         self._recurrent = recurrent

#         if recurrent:
#             self.gru = nn.GRU(recurrent_input_size, hidden_size)
#             for name, param in self.gru.named_parameters():
#                 if 'bias' in name:
#                     nn.init.constant_(param, 0)
#                 elif 'weight' in name:
#                     nn.init.orthogonal_(param)

#     @property
#     def is_recurrent(self):
#         return self._recurrent

#     @property
#     def recurrent_hidden_state_size(self):
#         if self._recurrent:
#             return self._hidden_size
#         return 1

#     @property
#     def output_size(self):
#         return self._hidden_size

#     def _forward_gru(self, x, hxs, masks):
#         if x.size(0) == hxs.size(0):

#             assert(x.ndim == 2 and hxs.ndim == 2)
#             x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
#             x = x.squeeze(0)
#             hxs = hxs.squeeze(0)
#         else:
#             raise Exception
#             # pdb.set_trace()
#             # # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
#             # N = hxs.size(0)
#             # T = int(x.size(0) / N)
#             #
#             # # unflatten
#             # x = x.view(T, N, x.size(1))
#             #
#             # # Same deal with masks
#             # masks = masks.view(T, N)
#             #
#             # # Let's figure out which steps in the sequence have a zero for any agent
#             # # We will always assume t=0 has a zero in it as that makes the logic cleaner
#             # has_zeros = ((masks[1:] == 0.0) \
#             #              .any(dim=-1)
#             #              .nonzero()
#             #              .squeeze()
#             #              .cpu())
#             #
#             # # +1 to correct the masks[1:]
#             # if has_zeros.dim() == 0:
#             #     # Deal with scalar
#             #     has_zeros = [has_zeros.item() + 1]
#             # else:
#             #     has_zeros = (has_zeros + 1).numpy().tolist()
#             #
#             # # add t=0 and t=T to the list
#             # has_zeros = [0] + has_zeros + [T]
#             #
#             # hxs = hxs.unsqueeze(0)
#             # outputs = []
#             # for i in range(len(has_zeros) - 1):
#             #     # We can now process steps that don't have any zeros in masks together!
#             #     # This is much faster
#             #     start_idx = has_zeros[i]
#             #     end_idx = has_zeros[i + 1]
#             #
#             #     rnn_scores, hxs = self.gru(
#             #         x[start_idx:end_idx],
#             #         hxs * masks[start_idx].view(1, -1, 1))
#             #
#             #     outputs.append(rnn_scores)
#             #
#             # # assert len(outputs) == T
#             # # x is a (T, N, -1) tensor
#             # x = torch.cat(outputs, dim=0)
#             # # flatten
#             # x = x.view(T * N, -1)
#             # hxs = hxs.squeeze(0)

#         return x, hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.lstm = nn.LSTM(recurrent_input_size, hidden_size)
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_lstm(self, x, hidden, masks):
        if x.size(0) == hidden[0].size(0):

            assert(x.ndim == 2 and hidden[0].ndim == 2)
            x, (h, c) = self.lstm(x.unsqueeze(0), ((hidden[0] * masks).unsqueeze(0), (hidden[1] * masks).unsqueeze(0)))
            x = x.squeeze(0)
            h = h.squeeze(0)
            c = c.squeeze(0)
        else:
            raise Exception

        return x, (h, c)


class GoalEncoder(nn.Module):
    def __init__(self, num_classes, output_dim, obj_class_encoder=None):
        super(GoalEncoder, self).__init__()


        if obj_class_encoder is None:
            inp_dim = output_dim
            self.object_embedding = nn.Embedding(num_classes, inp_dim)
        else:
            self.object_embedding = obj_class_encoder
            inp_dim = self.object_embedding.embedding_dim

        self.combine_obj_loc = nn.Sequential(
            nn.Linear(inp_dim*2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, object_class_name, loc_class_name, mask_goal_pred):
        obj_embedding = self.object_embedding(object_class_name)
        loc_embedding = self.object_embedding(loc_class_name)
        obj_loc = torch.cat([obj_embedding, loc_embedding], axis=2)
        object_location = self.combine_obj_loc(obj_loc)

        num_preds = mask_goal_pred.sum(-1)
        #norm_mask = (mask_goal_pred/num_preds.unsqueeze(-1)).unsqueeze(-1)
        # Difference with low level policy
        norm_mask = mask_goal_pred.unsqueeze(-1)

        average_pred = (object_location * norm_mask).sum(1)

        if torch.isnan(average_pred).any():
            pdb.set_trace()
        return average_pred

    # def forward(self, object_class_name, loc_class_name, mask_goal_pred):
    #     obj_embedding = self.object_embedding(object_class_name)
    #     loc_embedding = self.object_embedding(loc_class_name)
    #     obj_loc = torch.cat([obj_embedding, loc_embedding], axis=2)
    #     object_location = self.combine_obj_loc(obj_loc)
        
    #     sum_pred = (object_location * mask_goal_pred).sum(1)

    #     if torch.isnan(sum_pred).any():
    #         pdb.set_trace()
    #     return sum_pred


# class GoalAttentionModel(NNBase):
#     def __init__(self, recurrent=False, hidden_size=128, num_classes=100, node_encoder=None, context_type='avg'):
#         super(GoalAttentionModel, self).__init__(recurrent, hidden_size, hidden_size)

#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0), nn.init.calculate_gain('relu'))


#         self.main = node_encoder
#         self.context_size = hidden_size
#         self.critic_linear = init_(nn.Linear(hidden_size, 1))

#         self.object_context_combine = self.mlp2l(2 * hidden_size, hidden_size)

#         self.goal_encoder = GoalEncoder(num_classes, 2 * hidden_size, obj_class_encoder=self.main.object_class_encoding)
#         # self.goal_encoder = nn.EmbeddingBag(num_classes, hidden_size, mode='sum')
#         self.context_type = context_type

#         self.fc_att_action = self.mlp2l(hidden_size * 2, hidden_size)
#         self.fc_att_object = self.mlp2l(hidden_size * 2, hidden_size)
#         self.train()

#     def mlp2l(self, dim_in, dim_out):
#         return nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out))

#     def forward(self, inputs, rnn_hxs, masks):
#         # Use transformer to get feats for every object
#         mask_visible = inputs['mask_object']

#         features_obj = self.main(inputs)
#         #pdb.set_trace()

#         # 1 x ndim. Avg pool the features for the context vec
#         mask_visible = mask_visible.unsqueeze(-1)

#         # Mean pool of transformer
#         if self.context_type == 'avg':
#             context_vec = (features_obj * mask_visible).sum(1) / (1e-9 + mask_visible.sum(1))
#         else:
#             context_vec = features_obj[:, 0, :]


#         # Goal embedding
#         obj_class_name = inputs['target_obj_class']  # [:, 0].long()
#         loc_class_name = inputs['target_loc_class']  # [:, 0].long()
#         mask_goal = inputs['mask_goal_pred']

#         goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)

#         # goal_encoding_obj = self.goal_encoder(obj_class_name).squeeze(1)
#         # goal_encoding_loc = self.goal_encoder(loc_class_name).squeeze(1)
#         # goal_encoding = torch.cat([goal_encoding_obj, goal_encoding_loc], dim=-1)


#         goal_mask_action = torch.sigmoid(self.fc_att_action(goal_encoding))
#         goal_mask_object = torch.sigmoid(self.fc_att_object(goal_encoding))

#         # Recurrent context
#         if self.is_recurrent:
#             r_context_vec, rnn_hxs = self._forward_gru(context_vec, rnn_hxs, masks)
#         else:
#             r_context_vec = context_vec

#         # h' = GA . h [bs, h]
#         context_goal = goal_mask_action * r_context_vec

#         # Combine object representations with global representations
#         r_object_vec = torch.cat([features_obj, r_context_vec.unsqueeze(1).repeat(1, features_obj.shape[1], 1)], 2)
#         r_object_vec_comb = self.object_context_combine(r_object_vec)

#         # Sg' = GA . Sg [bs, N, h]
#         object_goal = goal_mask_object[:, None, :] * r_object_vec_comb

#         if torch.isnan(context_goal).any() or torch.isnan(object_goal).any():
#             pdb.set_trace()

#         return context_goal, object_goal, rnn_hxs


class GoalAttentionModel(NNBase):
    def __init__(self, recurrent=False, hidden_size=128, num_classes=100, node_encoder=None, context_type='avg'):
        super(GoalAttentionModel, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))


        self.main = node_encoder
        self.context_size = hidden_size
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.object_context_combine = self.mlp2l(2 * hidden_size, hidden_size)

        self.goal_encoder = GoalEncoder(num_classes, 2 * hidden_size)#, obj_class_encoder=self.main.object_class_encoding)
        # self.goal_encoder = nn.EmbeddingBag(num_classes, hidden_size, mode='sum')
        self.context_type = context_type

        self.fc_att_action = self.mlp2l(hidden_size * 2, hidden_size)
        self.fc_att_object = self.mlp2l(hidden_size * 2, hidden_size)
        self.train()

    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out))

    def forward(self, inputs, rnn_hxs, masks):
        # Use transformer to get feats for every object
        mask_visible = inputs['mask_object']

        features_obj = self.main(inputs)
        #pdb.set_trace()

        # 1 x ndim. Avg pool the features for the context vec
        mask_visible = mask_visible.unsqueeze(-1)

        # Mean pool of transformer
        if self.context_type == 'avg':
            context_vec = (features_obj * mask_visible).sum(1) / (1e-9 + mask_visible.sum(1))
        else:
            context_vec = features_obj[:, 0, :]


        # Goal embedding
        obj_class_name = inputs['target_obj_class']  # [:, 0].long()
        loc_class_name = inputs['target_loc_class']  # [:, 0].long()
        mask_goal = inputs['mask_goal_pred']

        goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)

        # goal_encoding_obj = self.goal_encoder(obj_class_name).squeeze(1)
        # goal_encoding_loc = self.goal_encoder(loc_class_name).squeeze(1)
        # goal_encoding = torch.cat([goal_encoding_obj, goal_encoding_loc], dim=-1)


        goal_mask_action = torch.sigmoid(self.fc_att_action(goal_encoding))
        goal_mask_object = torch.sigmoid(self.fc_att_object(goal_encoding))

       # h' = GA . h [bs, h]
        context_goal = goal_mask_action * context_vec

        # Recurrent context
        if self.is_recurrent:
            r_context_vec, rnn_hxs = self._forward_lstm(context_goal, rnn_hxs, masks)
        else:
            r_context_vec = context_goal

        # Combine object representations with global representations
        r_object_vec = torch.cat([goal_mask_object[:, None, :] * features_obj, r_context_vec.unsqueeze(1).repeat(1, features_obj.shape[1], 1)], 2)
        r_object_vec_comb = self.object_context_combine(r_object_vec)

        if torch.isnan(r_context_vec).any() or torch.isnan(r_object_vec_comb).any():
            pdb.set_trace()

        return r_context_vec, r_object_vec_comb, rnn_hxs


class GraphEncoder(nn.Module):
    def __init__(self, hidden_size=128, max_nodes=100, num_rels=5, num_classes=100, num_states=4):
        super(GraphEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.graph_encoder = GraphModelGGNN(
            num_classes=num_classes, num_nodes=max_nodes, h_dim=hidden_size, out_dim=hidden_size, num_rels=num_rels, num_states=num_states)
        self.object_class_encoding = self.graph_encoder.class_encoding

    def forward(self, inputs):
        # Build the graph
        hidden_feats = self.graph_encoder(inputs)
        return hidden_feats



class TransformerBase(nn.Module):

    def __init__(self, hidden_size=128, max_nodes=150, num_classes=100, num_states=4):
        super(TransformerBase, self).__init__()

        self.main = Transformer(num_classes=num_classes, num_nodes=max_nodes, in_feat=hidden_size, out_feat=hidden_size)
        #self.single_object_encoding = ObjNameCoordEncode(output_dim=hidden_size, num_classes=num_classes)
        self.single_object_encoding = ObjNameCoordStateEncode(output_dim=hidden_size, num_classes=num_classes, num_states=num_states)
        self.object_class_encoding = self.single_object_encoding.class_embedding
        self.train()

    def forward(self, inputs):
        # Use transformer to get feats for every object
        mask_visible = inputs['mask_object']
        input_node_embedding = self.single_object_encoding(inputs['class_objects'].long(),
                                                           inputs['object_coords'],
                                                           inputs['states_objects']).squeeze(1)
        node_embedding = self.main(input_node_embedding, mask_visible)
        return node_embedding




class ObjNameCoordStateEncode(nn.Module):
    def __init__(self, output_dim=128, num_classes=50, num_states=4):
        super(ObjNameCoordStateEncode, self).__init__()
        assert output_dim % 2 == 0
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(num_classes, int(output_dim / 2))
        self.state_embedding = nn.Linear(num_states, int(output_dim / 2))
        self.coord_embedding = nn.Sequential(nn.Linear(6, int(output_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(output_dim / 2), int(output_dim / 2)))
        inp_dim = int(output_dim + output_dim/2)
        self.combine = nn.Sequential(nn.ReLU(), nn.Linear(inp_dim, output_dim))

    def forward(self, class_ids, coords, state):
        state_embedding = self.state_embedding(state)
        class_embedding = self.class_embedding(class_ids)
        coord_embedding = self.coord_embedding(coords)
        inp = torch.cat([class_embedding, coord_embedding, state_embedding], dim=2)

        return self.combine(inp)


class ObjNameCoordEncode(nn.Module):
    def __init__(self, output_dim=128, num_classes=50):
        super(ObjNameCoordEncode, self).__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(num_classes, int(output_dim / 2))
        self.coord_embedding = nn.Sequential(nn.Linear(3, int(output_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(output_dim / 2), int(output_dim / 2)))

    def forward(self, class_ids, coords):
        class_embedding = self.class_embedding(class_ids)
        coord_embedding = self.coord_embedding(coords)
        return torch.cat([class_embedding, coord_embedding], dim=2)



