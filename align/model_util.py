import copy
import gc
import math
import torch
import numpy as np

from autil import alignment


def gen_neg(ent_embed, tt_links_new, metric, neg_k):
    with torch.no_grad():
        es1 = list(tt_links_new[:, 0]) # [e1 for e1,e2 in tt_links_new]
        es2 = list(tt_links_new[:, 1]) # [e2 for e1,e2 in tt_links_new]

        neg1_array = gen_neg_each(ent_embed, es1, metric, neg_k)
        neg2_array = gen_neg_each(ent_embed, es2, metric, neg_k)

        neg_pair = []
        for i in range(len(es2)):
            e1, e2 = tt_links_new[i]
            e1_neg, e2_neg = list(neg1_array[i]), list(neg2_array[i])
            if e1 in e1_neg:
                e1_neg.remove(e1)
            if e2 in e1_neg:
                e1_neg.remove(e2)
            if e1 in e2_neg:
                e2_neg.remove(e1)
            if e2 in e2_neg:
                e2_neg.remove(e2)

            for j in range(neg_k):
                neg_pair.append((e1, e2, e1_neg[j], e2_neg[j]))

        neg_pair = torch.LongTensor(np.array(neg_pair))  # eer_adj_data
        if ent_embed.is_cuda:
            neg_pair = neg_pair.cuda()

        return neg_pair


def gen_neg_each(ent_embed, left_ents, metric, neg_k):
    min_index = alignment.torch_sim_min_topk(ent_embed[left_ents, :], ent_embed, top_num=neg_k + 2, metric=metric, isdetach=True)

    e_t = len(left_ents)
    neg = []
    for i in range(e_t):
        rank = min_index[i, :].tolist()
        neg.append(rank)

    neg = np.array(neg) # neg.reshape((e_t * self.neg_k,))
    return neg  # (n*k,)


################################
def link_inset(list1):
    list2 = []
    for i in list1:
        if i not in list2:
            list2.append(i)
    return list2


############################################

def noin_fun(test_links, test_candidates):
    noin_test_candidate_num = 0
    test_links_new = []
    firstin = 0
    for e1, e2 in test_links:
        if e2 == test_candidates[e1][0]:
            firstin += 1
        if e2 not in test_candidates[e1]:
            noin_test_candidate_num += 1
            #print(e1, e2)
        else:
            test_links_new.append((e1, e2))
    return noin_test_candidate_num, firstin
