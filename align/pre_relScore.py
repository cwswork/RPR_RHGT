import copy
import time
import torch
import numpy as np

from autil import alignment, fileUtil

# 对齐路径
def get_align_path(ent_embed, train_links, ent_neigh_dict, kg_E, kg_R):
    max_neighbors_num = len(max(ent_neigh_dict.values(), key=lambda x: len(x)))  # 最大邻居
    print('Maximum number of neighbors:' + str(max_neighbors_num))  # max_neighbors_num = 235
    ent_pad_id, rel_pad_id = kg_E, kg_R
    left_neigh_match = neigh_match(ent_embed, train_links, ent_neigh_dict,
                                                max_neighbors_num, ent_pad_id, rel_pad_id)
    RR_pair_dict, temp_RR_list, temp_notRR_list = rel_match(left_neigh_match, ent_pad_id, threshold_num=20) # 20
    return RR_pair_dict, temp_RR_list, temp_notRR_list


def get_align_rel(ent_embed, train_links, rel_triples, kg_E, kg_R, aligned_rnum=-1):

    ent_neigh_dict = dict()
    ent_neigh_num = 0
    for eid in range(kg_E):
        ent_neigh_dict[eid] = []
    for (h, r, t) in rel_triples: # 无方向
        ent_neigh_dict[h].append((t, r))
        ent_neigh_num += 1
        if h!=t:
            ent_neigh_dict[t].append((h, r))
            ent_neigh_num += 1

    max_neighbors_num = len(max(ent_neigh_dict.values(), key=lambda x: len(x)))
    print('Maximum number of neighbors:' + str(max_neighbors_num))
    ent_pad_id, rel_pad_id = kg_E, kg_R
    left_neigh_match = neigh_match(ent_embed, train_links, ent_neigh_dict,
                                                max_neighbors_num, ent_pad_id, rel_pad_id)

    RR_pair_dict, temp_RR_list, temp_notRR_list = rel_match(left_neigh_match, ent_pad_id, threshold_num=10, aligned_rnum=aligned_rnum)
    return RR_pair_dict, temp_RR_list, temp_notRR_list, ent_neigh_dict, ent_neigh_num


def get_align_rel_inverse(ent_embed, train_links, rel_triples, kg_E, kg_R, aligned_rnum=-1):

    ent_neigh_dict = dict()
    for eid in range(kg_E):
        ent_neigh_dict[eid] = []
    for (h, r, t) in rel_triples:
        ent_neigh_dict[h].append((t, r))
        ent_neigh_dict[t].append((h, r+kg_R))

    max_neighbors_num = len(max(ent_neigh_dict.values(), key=lambda x: len(x)))  # 最大邻居
    print('Maximum number of neighbors:' + str(max_neighbors_num))  # max_neighbors_num = 235
    ent_pad_id, rel_pad_id = kg_E, kg_R*2  # 空白邻居的实体和关系ID
    left_neigh_match = neigh_match(ent_embed, train_links, ent_neigh_dict,
                                                max_neighbors_num, ent_pad_id, rel_pad_id)
    RR_pair_dict, temp_RR_list, temp_notRR_list = rel_match(left_neigh_match, ent_pad_id, threshold_num=10, aligned_rnum=aligned_rnum)
    return RR_pair_dict, temp_RR_list, temp_notRR_list, ent_neigh_dict


def neigh_match(ename_embed, train_links_array, ent_neigh_dict_old, max_neighbors_num, ent_pad_id, rel_pad_id):
    """Similarity Score (between entity pairs)
        return: [B,neigh,5]: (tail_i, tail_j, rel_i, rel_j, ent_ij_sim)
    """
    ent_neigh_dict = copy.deepcopy(ent_neigh_dict_old)
    for e in ent_neigh_dict.keys():
        pad_list = [(ent_pad_id, rel_pad_id)] * (max_neighbors_num - len(ent_neigh_dict[e]))
        ent_neigh_dict[e] += pad_list

    if ename_embed.is_cuda:
        ename_embed = ename_embed.detach().cpu().numpy()
    else:
        ename_embed = ename_embed.detach().numpy()
    dim = len(ename_embed[0])
    zero_embed = [0.0 for _ in range(dim)]  # <PAD> embedding
    ename_embed = np.vstack([ename_embed, zero_embed])
    ename_embed = torch.FloatTensor(ename_embed)

    print("train_pairs (e1,e2) num is: {}".format(len(train_links_array)))
    start_time = time.time()
    left_neigh_match, right_neigh_match = [], []
    batch_size = 1000
    for start_pos in range(0, len(train_links_array), batch_size):  # len(ent_pairs)=750000
        end_pos = min(start_pos + batch_size, len(train_links_array))
        batch_ent_pairs = train_links_array[start_pos: end_pos]
        e1s = [e1 for e1, e2 in batch_ent_pairs]
        e2s = [e2 for e1, e2 in batch_ent_pairs]
        #(t, r)
        er1_neigh = np.array([ent_neigh_dict[e1] for e1 in e1s])  # size: [B(Batchsize),ne1(e1_neighbor_max_num)]
        er2_neigh = np.array([ent_neigh_dict[e2] for e2 in e2s])
        e1_neigh = er1_neigh[:, :, 0]  # e
        e2_neigh = er2_neigh[:, :, 0]
        r1_neigh = er1_neigh[:, :, 1]  # r
        r2_neigh = er2_neigh[:, :, 1]

        e1_neigh_tensor = torch.LongTensor(e1_neigh)  # [B,neigh]
        e2_neigh_tensor = torch.LongTensor(e2_neigh)
        e1_neigh_emb = ename_embed[e1_neigh_tensor]  # [B,neigh,embedding_dim]
        e2_neigh_emb = ename_embed[e2_neigh_tensor]

        sim_mat_max = alignment.cosine_similarity3(e1_neigh_emb, e2_neigh_emb)
        max_scoce, max_index = sim_mat_max.topk(k=1, dim=-1, largest=True)
        max_scoce = max_scoce.squeeze(-1)  # [B,neigh,1] -> [B,neigh] #get max value.
        max_index = max_index.squeeze(-1)

        batch_match = np.zeros([e1_neigh.shape[0], e1_neigh.shape[1], 5])
        for e in range(e1_neigh.shape[0]): # [B,neigh]
            e1_array = e1_neigh[e] # [neigh,1] = >[neigh]
            e2_array = e2_neigh[e, max_index[e]]
            r1_array = r1_neigh[e]
            r2_array = r2_neigh[e, max_index[e]]
            scoce_array = max_scoce[e]
            if len(e2_array) != len(set(e2_array.tolist())):
                for i in range(1, len(e2_array)):
                    if e1_array[i] == ent_pad_id:
                        break

                    if e2_array[i] in e2_array[0: i]:
                        index = np.where(e2_array[0: i] == e2_array[i])[0][0]
                        if scoce_array[index] > scoce_array[i]:
                            e2_array[i] = ent_pad_id
                        else:
                            e2_array[index] = ent_pad_id
            batch_match[e] = np.vstack((e1_array, e2_array, r1_array, r2_array, scoce_array)).T

        if type(left_neigh_match) is np.ndarray:
            left_neigh_match = np.vstack((left_neigh_match, batch_match))
        else:
            left_neigh_match = batch_match

    print("all ent pair left_neigh_match shape:{}".format(left_neigh_match.shape) )
    print("using time {:.3f}".format(time.time() - start_time))
    return left_neigh_match


def rel_match(left_neigh_match, ent_pad_id, threshold_num=10, aligned_rnum=-1):
    rel_pair_score = {}
    ent_ij_sim5_num = 0
    for neigh_ll in left_neigh_match.tolist():
        for (tail_i, tail_j, rel_i, rel_j, ent_ij_sim) in neigh_ll:
            tail_i, tail_j, rel_i, rel_j = int(tail_i), int(tail_j), int(rel_i), int(rel_j)
            if tail_i == ent_pad_id or tail_j == ent_pad_id: # or ent_ij_sim == 0.0
                continue

            if ent_ij_sim < 0.5: # 0.5 取值是cosine, 取值是[-1,1]， 排除相似度低的
                ent_ij_sim5_num += 1
                continue

            if (rel_i, rel_j) not in rel_pair_score.keys():
                rel_pair_score[(rel_i, rel_j)] = [ent_ij_sim, 1]
            else:
                rel_pair_score[(rel_i, rel_j)][0] += ent_ij_sim  # 相似度叠加
                rel_pair_score[(rel_i, rel_j)][1] += 1 # 数量

    for rel_pair, (score, num) in rel_pair_score.items():
        rel_pair_score[rel_pair] = [score/num, num]

    print('ent_ij_sim5_num:' + str(ent_ij_sim5_num))
    print("all rel_pair_score len:" + str(len(rel_pair_score)))
    sim_rank_order = sorted(rel_pair_score.items(), key=lambda kv: kv[1][1], reverse=True)
    RR_list, notRR_list = [], []
    RR_pair_dict = dict()
    for (r1_id, r2_id), (sim_v, num) in sim_rank_order:
        if r1_id not in RR_pair_dict.keys() and r2_id not in RR_pair_dict.values():
            if aligned_rnum != -1:
                if r1_id < aligned_rnum or r2_id < aligned_rnum:
                    continue

            if num < threshold_num:
                continue
            RR_pair_dict[r1_id] = r2_id
            RR_list.append([r1_id, r2_id, sim_v, num])
        else:
            notRR_list.append([r1_id, r2_id, sim_v, num])

    RR_list = sorted(RR_list, key=lambda kv: kv[0], reverse=False)
    notRR_list = sorted(notRR_list, key=lambda kv: kv[0], reverse=False)
    return RR_pair_dict, RR_list, notRR_list

def get_Rname_dict(datasetPath, kg_R):
    kg1_index2rel = fileUtil.load_ids2dict(datasetPath + 'kg1_rel_dict', read_kv='kv')  # id:name
    kg2_index2rel = fileUtil.load_ids2dict(datasetPath + 'kg2_rel_dict', read_kv='kv')

    kg1_name2id = dict()
    for rid1, rname1 in kg1_index2rel.items():
        rname1 = rname1.split('/')[-1]
        kg1_name2id[rname1] = rid1

    kg2_name2id = dict()
    for rid2, rname2 in kg2_index2rel.items():
        rname2 = rname2.split('/')[-1]
        kg2_name2id[rname2] = rid2

    Rname_dict = {}
    for rname1, rid1 in kg1_name2id.items():
        if rname1 in kg2_name2id.keys():
            Rname_dict[rid1] = kg2_name2id[rname1]

    return kg1_index2rel, kg2_index2rel, Rname_dict


def get_Rname_dict_inverse(datasetPath, kg_R):
    kg1_index2rel = fileUtil.load_ids2dict(datasetPath + 'kg1_rel_dict', read_kv='kv')
    kg2_index2rel = fileUtil.load_ids2dict(datasetPath + 'kg2_rel_dict', read_kv='kv')
    kg1_name2id = dict()
    for rid1, rname1 in kg1_index2rel.items():
        rname1 = rname1.split('/')[-1]
        kg1_name2id[rname1] = rid1

    kg2_name2id = dict()
    for rid2, rname2 in kg2_index2rel.items():
        rname2 = rname2.split('/')[-1]
        kg2_name2id[rname2] = rid2

    Rname_dict = {}
    for rname1, rid1 in kg1_name2id.items():
        if rname1 in kg2_name2id.keys():
            Rname_dict[rid1] = kg2_name2id[rname1]
    print('Rname_dict:', len(Rname_dict))
    # inverse
    newdict = dict()
    for rid1, rname1 in kg1_index2rel.items():
        newdict[rid1 + kg_R] = rname1
    kg1_index2rel.update(newdict)

    newdict = dict()
    for rid2, rname2 in kg2_index2rel.items():
        newdict[rid2 + kg_R] = rname2
    kg2_index2rel.update(newdict)

    newdict = dict()
    for rid1, rid2 in Rname_dict.items():
        newdict[rid1 + kg_R] = rid2 + kg_R
    Rname_dict.update(newdict)

    return kg1_index2rel, kg2_index2rel, Rname_dict


def reset_relid(RR_pair_dict, kg1_index2rel, kg2_index2rel, ent_neigh_dict):
    Rold2new_dict = dict()
    # alignment relation
    kg_R_new = 0
    for r1, r2 in RR_pair_dict.items():
        Rold2new_dict[r1] = kg_R_new
        Rold2new_dict[r2] = kg_R_new
        kg_R_new += 1
    align_Rnum = kg_R_new
    # KG1
    for rid_old in kg1_index2rel.keys():
        if rid_old not in Rold2new_dict.keys():
            Rold2new_dict[rid_old] = kg_R_new
            kg_R_new += 1

    # KG2
    for rid_old in kg2_index2rel.keys():
        if rid_old not in Rold2new_dict.keys():
            Rold2new_dict[rid_old] = kg_R_new
            kg_R_new += 1

    # update ent_neigh_dict
    align_rel_triple_num = 0
    align_ent_neigh_dict = dict()
    for e1, er_list in ent_neigh_dict.items():
        er_list_new = [(e2, Rold2new_dict[r]) for e2, r in er_list]
        ent_neigh_dict[e1] = er_list_new

        er_list_align = [(e2, new_r) for e2, new_r in er_list_new if new_r < align_Rnum]
        align_ent_neigh_dict[e1] = er_list_align
        align_rel_triple_num += len(er_list_align)

    return Rold2new_dict, ent_neigh_dict, align_ent_neigh_dict, align_rel_triple_num, kg_R_new, align_Rnum

