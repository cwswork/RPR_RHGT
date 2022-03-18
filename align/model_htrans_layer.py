import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
import math

from autil import alignment

# 多头Transformer
class Multi_Htrans_Layer(nn.Module):
    def __init__(self, ent_neigh_dict, kg_E, kg_R, config):
        super(Multi_Htrans_Layer, self).__init__()
        if config.is_cuda:
            self.device = config.device
        self.n_layers = config.n_layers

        self.kg_E = kg_E
        self.rel_adj, self.rel_edge_mat = self.set_Radj(kg_E, kg_R, ent_neigh_dict, config.is_cuda)
        # 层次比例
        self.rel_skip_w = nn.Parameter(torch.ones(self.n_layers))
        self.model_layers = nn.ModuleList()
        #self.n_heads = config.n_heads
        for l in range(config.n_heads):
            rel_layer = Htrans_Layer(kg_E, kg_R, config)
            self.model_layers.append(rel_layer)


    def set_Radj(self, kg_E, kg_R, ent_neigh_dict, is_cuda):
        r_head_array = np.zeros((kg_R, kg_E))
        r_tail_array = np.zeros((kg_R, kg_E))
        for h, trlist in ent_neigh_dict.items():
            for t, r in trlist:
                r_head_array[r][h] += 1
                r_tail_array[r][t] += 1
        # 头尾节点-关系矩阵：r_head:array[r,h]=1,  r_tail:array[r,t]=1
        r_head = torch.FloatTensor(r_head_array)
        r_tail = torch.FloatTensor(r_tail_array)

        # 辅助信息
        r_head_sum = torch.unsqueeze(torch.sum(r_head, dim=-1), -1)  # (R,E)
        r_tail_sum = torch.unsqueeze(torch.sum(r_tail, dim=-1), -1)  # (R,E)
        r_head_sum = torch.where(r_head_sum == 0, torch.tensor(0.), 1. / r_head_sum)  # Instead of countdown
        r_tail_sum = torch.where(r_tail_sum == 0, torch.tensor(0.), 1. / r_tail_sum)  # Instead of countdown

        if is_cuda:
            r_head_sum = r_head_sum.cuda(self.device)
            r_tail_sum = r_tail_sum.cuda(self.device)
            r_head = r_head.cuda(self.device)
            r_tail = r_tail.cuda(self.device)

        rel_adj = [r_head, r_head_sum, r_tail, r_tail_sum]

        ##
        edge_index_ht = []
        edge_type_r = []
        for h in range(self.kg_E):
            for t, r in ent_neigh_dict[h]:  # trlist
                edge_index_ht.append((h, t))  # [target, source]
                edge_type_r.append(r)
        edge_index_ht = torch.LongTensor(edge_index_ht).t()  # [2, edge_len]
        edge_type_r = torch.LongTensor(edge_type_r)  # 浮点数

        if is_cuda:
            edge_index_ht = edge_index_ht.cuda(self.device)
            edge_type_r = edge_type_r.cuda(self.device)
        rel_edge_mat = [edge_index_ht, edge_type_r]
        return rel_adj, rel_edge_mat


    def forward(self, ent_name, ent_embed_in):
        ent_out = ent_name #ent_embed_in
        for l in range(self.n_layers):
            if len(self.model_layers) == 1:
                ent_out_head = self.model_layers[0](ent_out, self.rel_adj, self.rel_edge_mat)
            else:
                # Multi_head
                embed_out_list = []
                for i in range(len(self.model_layers)):
                    out_embed = self.model_layers[i](ent_out, self.rel_adj, self.rel_edge_mat)
                    embed_out_list.append(out_embed)
                ent_out_head = torch.cat(embed_out_list, dim=-1)

            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            alpha = torch.sigmoid(self.rel_skip_w[l])
            ent_out = ent_out_head * alpha + ent_embed_in * (1 - alpha)
            #ent_out = F.normalize(ent_out, p=1, dim=-1)


        return ent_out, self.rel_skip_w.detach().tolist() #+ [alpha.detach().tolist()]

    # add a highway layer
    def highway__(self, e_layer1, e_layer2):
        # (E,dim) * (dim,dim)
        transform_gate = torch.mm(e_layer1, self.highwayWr) + self.highwaybr.squeeze(1)
        transform_gate = torch.sigmoid(transform_gate)
        e_layer = transform_gate * e_layer2 + (1.0 - transform_gate) * e_layer1

        return e_layer

class Htrans_Layer(MessagePassing):
    def __init__(self, kg_E, kg_R, config, **kwargs):
        super(Htrans_Layer, self).__init__(node_dim=0, aggr='add', **kwargs) #add， mean, max。

        # Super Parameter
        self.relu = nn.ReLU(inplace=True) # self.relu -> F.gelu
        self.dropout = nn.Dropout(config.dropout)
        self.leakyrelu = nn.LeakyReLU(config.LeakyReLU_alpha)  # LeakyReLU_alpha: leakyrelu

        # Relation
        self.kg_R = kg_R
        self.in_dim = 300 # config.e_dim #
        self.out_dim = config.e_dim
        self.dk_dim = self.out_dim // config.n_heads
        self.r_dim = self.dk_dim
        self.sqrt_dk = math.sqrt(self.dk_dim)

        self.k_linear_head = nn.Linear(self.in_dim, self.dk_dim)  # [in_dim, out_dim] = [in_dim, head_d]
        self.q_linear_head = nn.Linear(self.in_dim, self.dk_dim)
        self.v_linear_head = nn.Linear(self.in_dim, self.dk_dim)
        self.out_linear = nn.Linear(self.dk_dim*3, self.dk_dim)

        # relation
        self.attr_r = nn.Parameter(torch.ones(self.dk_dim*2, 1))

        self.relation_Left = nn.Parameter(torch.zeros(size=(self.in_dim, self.r_dim))) # W_r
        self.relation_Right = nn.Parameter(torch.zeros(size=(self.in_dim, self.r_dim)))

        weight_init_list = [self.relation_Left, self.relation_Right]
        for w in weight_init_list:
            glorot(w)

    def forward(self, ent_embed_in, rel_adj_list, rel_edge_mat):
        # rel embed
        rel_embed = self.get_rlayer(ent_embed_in, rel_adj_list)

        edge_index_ht, edge_type_r = rel_edge_mat
        ent_embed_out = self.propagate(edge_index_ht, ent_embed_in=ent_embed_in, edge_type_r=edge_type_r,
                                       rel_embed=rel_embed)
        return ent_embed_out


    def get_rlayer(self, ent_embed, rel_adj_list):
        [r_head, r_head_sum, r_tail, r_tail_sum] = rel_adj_list
        L_e_inlayer = torch.mm(ent_embed, self.relation_Left)
        L_r_embed = torch.matmul(r_head, L_e_inlayer)  # (R,E)*(E,d) => (R,d)
        L_r_embed = L_r_embed * r_head_sum  # / r_head_sum =>

        R_e_inlayer = torch.mm(ent_embed, self.relation_Right)
        R_r_embed = torch.matmul(r_tail, R_e_inlayer)  # (R,E)*(E,d) => (R,d)
        R_r_embed = R_r_embed * r_tail_sum  # / r_tail_sum =>

        r_embed = torch.cat([L_r_embed, R_r_embed], dim=-1)  # (r,600)
        r_embed = self.relu(r_embed)

        return r_embed


    def message(self, edge_index_i, ent_embed_in_i, ent_embed_in_j, edge_type_r, rel_embed):
        '''
            j: source, i: target; i <- j
            ent_embed： i,j
        '''
        ### Step 1: Heterogeneous Mutual Attention
        # k_linears_head(s), q_linears_head(t)  v_linear_head(s)
        k_mat = self.k_linear_head(ent_embed_in_j) # [T, d]
        q_mat = self.q_linear_head(ent_embed_in_i)

        # K(s)W(r) = K(h)*rel_mat(r)
        kq_mat = torch.cat([k_mat, q_mat], dim=-1) #  [T, 2d]
        r_mat = rel_embed[edge_type_r]  # [R, 2d] -> [T, 2d]
        ent_att = kq_mat * r_mat  # [T, 2d]
        # (K(s)//Q(t)) . R(r) * attr
        ent_att = torch.matmul(ent_att, self.attr_r) / self.sqrt_dk  # [T, 2d] * [2d, 1] -> [T, 1]
        ent_att = -self.leakyrelu(ent_att)

        # Step 2: Heterogeneous Message Passing
        v_mat = self.v_linear_head(ent_embed_in_j)  # -> [T, d]
        ent_msg = torch.cat([v_mat, r_mat], dim=-1)  # -> [T, 3d]

        # Step 3: # Attention(h,r,t) * Message(h,r,t)
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        ent_att = softmax(ent_att, edge_index_i)  # [T, 1]
        aggr_out = ent_att * ent_msg  # -> [T,1] . [T, 3d]
        del ent_msg, ent_att
        return aggr_out # [T, 3d]


    def update(self, aggr_out, ent_embed_in):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        aggr_out = self.relu(aggr_out) #[E, 2d]
        trans_out = self.dropout(self.out_linear(aggr_out)) # [E, 3d] -> [E, d]

        return trans_out



