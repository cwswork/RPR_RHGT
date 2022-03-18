import math

import numpy as np
import torch
import torch.nn as nn

from align import model_util
from autil import alignment
from align.model_htrans_layer import Multi_Htrans_Layer


class Align_Htrans(nn.Module):
    def __init__(self, kgs_data, config):
        super(Align_Htrans, self).__init__()
        self.myprint = config.myprint
        self.is_cuda = config.is_cuda
        if config.is_cuda:
            self.device = config.device
        self.metric = 'L1'
        self.l_beta = config.l_beta
        self.gamma_rel = config.gamma_rel
        self.neg_k = config.neg_k
        self.top_k = config.top_k

        # Super Parameter
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout)

        self.kg_E = kgs_data.kg_E
        self.kg_R = kgs_data.kg_R
        self.e_dim = config.e_dim
        # name embedding ##############
        self.kg_name_embed = kgs_data.ename_embed
        self.kg_name_model = nn.Linear(300, self.e_dim)


        self.model_rels = Multi_Htrans_Layer(kgs_data.ent_neigh_dict, kgs_data.kg_E, kgs_data.kg_R, config)
        if 'path' in config.model_type:
            self.isPath = True
            self.myprint('new kg_path:' + str(kgs_data.kg_path)) # self
            self.model_paths = Multi_Htrans_Layer(kgs_data.path_neigh_dict, kgs_data.kg_E, kgs_data.kg_path, config)
        else:
            self.isPath = False

        if self.is_cuda:
            self.kg_name_embed = self.kg_name_embed.cuda(self.device)
            self.model_rels = self.model_rels.cuda(self.device)
            if self.isPath:
                self.model_paths = self.model_paths.cuda(self.device)

        ##  ##############
        params = list(self.parameters())
        self.model_params = [{'params': params}]
        params_list = [param for param in self.state_dict()]
        self.myprint('model中所有参数名:{}\n{}'.format(str(len(params_list)), params_list.__str__()))

        ########################
        self.reNeg = False
        self.train_links_list = kgs_data.train_links
        self.notrain_links_list = kgs_data.test_links + kgs_data.valid_links
        self.train_links = np.array(kgs_data.train_links)
        self.test_links = np.array(kgs_data.test_links)
        if kgs_data.valid_links !=[]:
            self.valid_links = np.array(kgs_data.valid_links)
        else:
            self.valid_links = []

        self.left_non_train, self.right_non_train = [], []
        for e1, e2 in self.notrain_links_list:
            self.left_non_train.append(e1)
            self.right_non_train.append(e2)


    def resetPath(self, path_neigh_dict, kg_path):
        add_rid = kg_path
        for eid in range(self.kg_E):
            if len(path_neigh_dict[eid]) <1:
                path_neigh_dict[eid].append((eid, add_rid))

        return path_neigh_dict, kg_path+1

    # 2 rel_gat
    def forward(self):
        ###1 model_name
        end_embed_in = self.kg_name_model(self.kg_name_embed)
        ### 2 rel model
        rel_embed, skip_w = self.model_rels(self.kg_name_embed, end_embed_in)
        ###3 path model
        if self.isPath:
            path_embed, p_skip_w = self.model_paths(self.kg_name_embed, end_embed_in)
            skip_w_all = skip_w + p_skip_w
        else:
            path_embed = None
            skip_w_all = skip_w

        self.rel_embed, self.path_embed = rel_embed, path_embed
        skip_w_all = [round(w, 4) for w in skip_w_all]

        return str(skip_w_all)

    def accTest(self, tt_links_tensor, tt_links):

        Left_vec = self.rel_embed[tt_links_tensor[:, 0], :]
        Right_vec = self.rel_embed[tt_links_tensor[:, 1], :]
        Left_re = alignment.get_hits(Left_vec, Right_vec, tt_links, self.top_k, self.metric, LeftRight='Left')
        print('Rel:', Left_re[1])

        Left_vec = self.path_embed[tt_links_tensor[:, 0], :]
        Right_vec = self.path_embed[tt_links_tensor[:, 1], :]
        Left_re = alignment.get_hits(Left_vec, Right_vec, tt_links, self.top_k, self.metric, LeftRight='Left')
        print('Rel:', Left_re[1])


    def regen_neg(self):
        # neg gen
        self.train_neg_pairs_rel = model_util.gen_neg(self.rel_embed, self.train_links, self.metric, self.neg_k)
        if self.isPath:
            self.train_neg_pairs_path = model_util.gen_neg(self.path_embed, self.train_links, self.metric, self.neg_k)
        if self.is_cuda:
            self.train_neg_pairs_rel = self.train_neg_pairs_rel.cuda(self.device)
            if self.isPath:
                self.train_neg_pairs_path = self.train_neg_pairs_path.cuda(self.device)

        if type(self.valid_links) is np.ndarray:
            self.valid_neg_pairs_rel = model_util.gen_neg(self.rel_embed, self.valid_links, self.metric, self.neg_k)
            if self.isPath:
                self.valid_neg_pairs_path = model_util.gen_neg(self.path_embed, self.valid_links, self.metric, self.neg_k)
            if self.is_cuda:
                self.valid_neg_pairs_rel = self.valid_neg_pairs_rel.cuda(self.device)
                if self.isPath:
                    self.valid_neg_pairs_path = self.valid_neg_pairs_path.cuda(self.device)

        self.reNeg = False

    ## loss function
    def get_loss(self, epochs_i, link_type=1):
        # neg
        if (self.reNeg or epochs_i % 20 == 0) and link_type==1:
            self.regen_neg()

        if link_type == 1:
            tt_neg_rel = self.train_neg_pairs_rel
            if self.isPath:
                tt_neg_path = self.train_neg_pairs_path
        else:
            tt_neg_rel = self.valid_neg_pairs_rel
            if self.isPath:
                tt_neg_path = self.valid_neg_pairs_path

        loss = self.get_loss_each(self.rel_embed, tt_neg_rel)
        if self.isPath:
            loss2 = self.get_loss_each(self.path_embed, tt_neg_path)
            loss = loss + self.l_beta * loss2
        return loss


    def get_loss_each(self, e_embed, tt_neg_pairs):
        # loss
        pe1_embed = e_embed[tt_neg_pairs[:, 0]]
        pe2_embed = e_embed[tt_neg_pairs[:, 1]]
        A = alignment.mypair_distance_min(pe1_embed, pe2_embed, distance_type=self.metric)
        D = (A + self.gamma_rel)  # .view(t, 1)

        ne1_embed = e_embed[tt_neg_pairs[:, 2]]
        B = alignment.mypair_distance_min(pe1_embed, ne1_embed, distance_type=self.metric)
        loss1 = self.relu(D - B)  # (t, 50).view(t, -1)

        ne2_embed = e_embed[tt_neg_pairs[:, 3]]
        C = alignment.mypair_distance_min(pe2_embed, ne2_embed, distance_type=self.metric)
        loss2 = self.relu(D - C)

        loss = torch.mean(loss1 + loss2)  # torch.mean(loss)  sum
        return loss


    ## accuracy
    def accuracy(self, link_type=1):
        with torch.no_grad():
            if link_type == 1:
                tt_links = self.train_links
            elif link_type == 2:
                tt_links = self.valid_links
            elif link_type == 3:
                tt_links = self.test_links
            tt_links_tensor = torch.LongTensor(tt_links)

            if self.isPath:
                outEmbed = torch.cat((self.rel_embed, self.path_embed), dim=1)
                Left_vec = outEmbed[tt_links_tensor[:, 0], :]
                Right_vec = outEmbed[tt_links_tensor[:, 1], :]
            else:
                Left_vec = self.rel_embed[tt_links_tensor[:, 0], :]
                Right_vec = self.rel_embed[tt_links_tensor[:, 1], :]

            # alignment
            Left_re = alignment.get_hits(Left_vec, Right_vec, tt_links, self.top_k, self.metric, LeftRight='Left')
            # From Right
            if link_type == 3:
                Right_re = alignment.get_hits(Right_vec, Left_vec, tt_links[:, [1, 0]], self.top_k, self.metric, LeftRight='Right')
            else:
                Right_re = None

        return Left_re, Right_re

