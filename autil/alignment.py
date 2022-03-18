import gc
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import functional

##accuracy#
def get_hits(Left_vec, Right_vec, tt_links, top_k, metric, LeftRight='Left'):
    #t_begin = time.time()
    min_index, min_mat = torch_sim_min_topk_mat(Left_vec, Right_vec, top_num=top_k[-1], metric=metric)

    # left
    mr = 0
    mrr = 0
    tt_num = len(tt_links)
    all_hits = [0] * len(top_k)
    Hits_list = list()
    noHits1_num = 0
    # From left
    for row_i in range(min_index.shape[0]):
        e2_ranks_index = min_index[row_i, :].tolist()
        if row_i in e2_ranks_index:
            rank_index = e2_ranks_index.index(row_i)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    all_hits[j] += 1
        else:
            row_mat_index = min_mat[row_i].argsort().tolist()
            rank_index = row_mat_index.index(row_i)
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)

        e1_gold, e2_gold = tt_links[row_i]
        hits1_e2 = tt_links[e2_ranks_index[0], 1]
        Hits_list.append((LeftRight, e1_gold, e2_gold, hits1_e2, rank_index))
        if rank_index != 0:
            noHits1_num += 1

    assert len(Hits_list) == tt_num
    all_hits = [round(hh / tt_num * 100, 4) for hh in all_hits]
    mr /= tt_num
    mrr /= tt_num
    # From left
    result_str1 = "hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, noHits1:{}".format(
        top_k, all_hits, mr, mrr, noHits1_num)
    #print("get_hits_each: {:.6f}s".format(time.time() - t_begin))
    return [all_hits[0], result_str1, Hits_list] #Left_re


def get_hits_simple(Left_vec, Right_vec, tt_links, top_k, metric, LeftRight='Left'):
    t_begin = time.time()
    min_index, min_mat = torch_sim_min_topk_mat(Left_vec, Right_vec, top_num=top_k[-1], metric=metric)

    # left
    mr = 0
    mrr = 0
    tt_num = len(tt_links)
    all_hits = [0] * len(top_k)
    # From left
    for row_i in range(min_index.shape[0]):
        e2_ranks_index = min_index[row_i, :].tolist()
        if row_i in e2_ranks_index:
            rank_index = e2_ranks_index.index(row_i)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    all_hits[j] += 1
        else:
            row_mat_index = min_mat[row_i].argsort().tolist()
            rank_index = row_mat_index.index(row_i)
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        e1_gold, e2_gold = tt_links[row_i]
        hits1_e2 = tt_links[e2_ranks_index[0], 1]

    all_hits = [round(hh / tt_num * 100, 4) for hh in all_hits]
    mr /= tt_num
    mrr /= tt_num
    # From left
    result_str1 = "hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, cost time-{:.4f}s".format(
        top_k, all_hits, mr, mrr, time.time() - t_begin)
    return [all_hits[0], result_str1, []] #Left_re


##distance##################
def mypair_distance_min(a, b, distance_type="L1"):
    if distance_type == "L1":
        return functional.pairwise_distance(a, b, p=1)  # [B*C] 相似度高，值低
    elif distance_type == "L2":
        return functional.pairwise_distance(a, b, p=2)
    elif distance_type == "L2squared":
        return torch.pow(functional.pairwise_distance(a, b, p=2), 2)
    elif distance_type == "cosine":
        return 1 - torch.cosine_similarity(a, b)  # [B*C] 相似度高，值高(取负)


##similarity, topk########
def torch_sim_min_topk(embed1, embed2, top_num, metric='manhattan', isdetach=True, left_batch_size=None, right_batch_size=None):
    if left_batch_size == None:
        if embed1.is_cuda:
            left_batch_size = 1000
        else:
            left_batch_size = 5000

    if embed1.shape[0] <= left_batch_size:
        min_scoce, min_index, _ = torch_sim_min_topk_batch(embed1, embed2, top_num, metric=metric)
    else:
        links_len = embed1.shape[0]
        min_index_list = []
        for i in np.arange(0, links_len, left_batch_size):
            end = min(i + left_batch_size, links_len)
            min_index_batch, _ = torch_sim_min_topk_vseg(embed1[i:end, :], embed2, top_num, metric=metric, right_batch_size=right_batch_size)
            min_index_list.append(min_index_batch)

        min_index = torch.cat(min_index_list, 0)
        del min_index_list, min_index_batch
        gc.collect()

    if isdetach:
        min_index = min_index.int()
        if min_index.is_cuda:
            min_index = min_index.detach().cpu().numpy()
        else:
            min_index = min_index.detach().numpy()

    return min_index

def torch_sim_min_topk_mat(embed1, embed2, top_num, metric='manhattan', isdetach=True, left_batch_size=None, right_batch_size=None):
    if left_batch_size == None:
        if embed1.is_cuda:
            left_batch_size = 1000
        else:
            left_batch_size = 5000

    if embed1.shape[0] <= left_batch_size:
        min_scoce, min_index, min_mat = torch_sim_min_topk_batch(embed1, embed2, top_num, metric=metric, ismat=True)
    else:
        links_len = embed1.shape[0]
        min_index_list, min_mat_list = [], []
        for i in np.arange(0, links_len, left_batch_size):
            end = min(i + left_batch_size, links_len)
            min_index_batch, min_mat_batch = torch_sim_min_topk_vseg(embed1[i:end, :], embed2, top_num, metric=metric,
                                                 right_batch_size=right_batch_size, ismat=True)
            min_index_list.append(min_index_batch)
            min_mat_list.append(min_mat_batch)

        min_index = torch.cat(min_index_list, 0)
        min_mat = torch.cat(min_mat_list, 0)
        del min_index_list, min_index_batch, min_mat_list, min_mat_batch
        gc.collect()

    if isdetach:
        min_index = min_index.int()
        if min_index.is_cuda:
            min_index = min_index.detach().cpu().numpy()
        else:
            min_index = min_index.detach().numpy()
        if min_mat.is_cuda:
            min_mat = min_mat.detach().cpu().numpy()
        else:
            min_mat = min_mat.detach().numpy()
    return min_index, min_mat


def torch_sim_min_topk_s(embed1, embed2, top_num, metric='manhattan', isdetach=True, left_batch_size=None, right_batch_size=None):
    if left_batch_size == None:
        if embed1.is_cuda:
            left_batch_size = 1000
        else:
            left_batch_size = 5000

    if embed1.shape[0] <= left_batch_size:
        min_scoce, min_index, _ = torch_sim_min_topk_batch(embed1, embed2, top_num, metric=metric)
    else:
        links_len = embed1.shape[0]
        min_scoce_list, min_index_list = [], []
        for i in np.arange(0, links_len, left_batch_size):
            end = min(i + left_batch_size, links_len)
            min_scoce_batch, min_index_batch, _ = torch_sim_min_topk_vseg(embed1[i:end, :], embed2, top_num, metric=metric,
                                                  right_batch_size=right_batch_size, is_scoce=True)
            min_scoce_list.append(min_scoce_batch)
            min_index_list.append(min_index_batch)

        min_scoce = torch.cat(min_scoce_list, 0)
        min_index = torch.cat(min_index_list, 0)
        del min_index_list, min_scoce_list

    min_index = min_index.int()
    if isdetach:
        if min_index.is_cuda:
            min_scoce = min_scoce.detach().cpu().numpy()
            min_index = min_index.detach().cpu().numpy()
        else:
            min_scoce = min_scoce.detach().numpy()
            min_index = min_index.detach().numpy()

    gc.collect()
    return min_scoce, min_index

def torch_sim_min_topk_vseg(embed1, embed2, top_num, metric, ismat=False, right_batch_size=None, is_scoce=False):
    if right_batch_size == None:
        right_batch_size = 12500

    if embed2.shape[0] <= right_batch_size:
        min_scoce_merge, min_index_merge, min_mat_merge = torch_sim_min_topk_batch(embed1, embed2, top_num, metric=metric, ismat=ismat)
    else:
        right_len = embed2.shape[0]
        min_scoce_list, min_index_list, min_mat_list = [], [], []
        for beg_index in np.arange(0, right_len, right_batch_size):
            end = min(beg_index + right_batch_size, right_len)
            min_scoce_batch, min_index_batch, sim_mat_batch = torch_sim_min_topk_batch(embed1, embed2[beg_index:end, :], top_num, metric=metric, ismat=ismat)
            if beg_index != 0:
                min_index_batch += beg_index

            min_scoce_list.append(min_scoce_batch)
            min_index_list.append(min_index_batch)
            if ismat:
                min_mat_list.append(sim_mat_batch)
        min_scoce_merge = torch.cat(min_scoce_list, 1)
        min_index_merge = torch.cat(min_index_list, 1)
        if ismat:
            min_mat_merge = torch.cat(min_mat_list, 1)
        else:
            min_mat_merge = None
        del min_scoce_list,min_index_list

    top_index = min_scoce_merge.argsort(dim=-1, descending=False)
    top_index = top_index[:, :top_num]
    #
    row_count = embed1.shape[0]
    min_index = torch.zeros((row_count, top_num), )
    if is_scoce: min_scoce = torch.zeros((row_count, top_num))

    for i in range(row_count):
        min_index[i] = min_index_merge[i, top_index[i]]
        if is_scoce: min_scoce[i] = min_scoce_merge[i, top_index[i]]

    if is_scoce:
        return min_scoce, min_index, min_mat_merge
    else:
        return min_index, min_mat_merge

def torch_sim_min_topk_batch(embed1, embed2, top_num, metric='manhattan', ismat=False):
    if metric == 'L1' or metric == 'manhattan':  # L1 Manhattan
        sim_mat = torch.cdist(embed1, embed2, p=1.0)
    elif metric == 'L2' or metric == 'euclidean':  # L2 euclidean
        sim_mat = torch.cdist(embed1, embed2, p=2.0)
    elif metric == 'cosine': #  cosine
        sim_mat = 1 - cosine_similarity3(embed1, embed2)

    if len(embed2) > top_num:
        min_scoce_batch, min_index_batch = sim_mat.topk(k=top_num, dim=-1, largest=False)
    else:
        min_scoce_batch, min_index_batch = sim_mat.topk(k=len(embed2), dim=-1, largest=False)

    if ismat:
        return min_scoce_batch, min_index_batch, sim_mat
    else:
        del sim_mat
        gc.collect()
        return min_scoce_batch, min_index_batch, None

def sim_min_batch(embed1, embed2, metric='manhattan'):
    ''' 求两组实体列表的相似度， 越相似，值越大=》越相似，值越小 '''
    if metric == 'L1' or metric == 'manhattan':  # L1 Manhattan
        sim_mat = torch.cdist(embed1, embed2, p=1.0)
    elif metric == 'L2' or metric == 'euclidean':  # L2 euclidean
        sim_mat = torch.cdist(embed1, embed2, p=2.0)
    elif metric == 'cosine': #  cosine
        sim_mat = 1 - cosine_similarity3(embed1, embed2)  # [batch, net1, net1]

    return sim_mat


def cosine_similarity3(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    # a = a / torch.clamp(a.norm(dim=-1, keepdim=True, p=2), min=1e-5)
    # b = b / torch.clamp(b.norm(dim=-1, keepdim=True, p=2), min=1e-5)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    if len(b.shape) == 3:
        sim = torch.bmm(a, torch.transpose(b, 1, 2))
    else:
        sim = torch.mm(a, b.t())

    return sim


############


def task_divide(idx, n):
    '''  '''
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def divide_list(idx_list, batch_size):
    '''  '''
    total = len(idx_list)
    if batch_size <= 0 or total <= batch_size:
        return [idx_list]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            batchs_list.append(idx_list[beg:beg + batch_size])

        if beg + batch_size < total:
            batchs_list.append(idx_list[beg + batch_size:])
        return batchs_list

def divide_range(total, batch_size):
    '''  '''
    if batch_size <= 0 or total <= batch_size:
        return [0, total]
    else:
        n = total // batch_size +1
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            batchs_list.append(beg)
        end = beg + batch_size
        if end > total:
            end = total
        batchs_list.append(end)
        return batchs_list

def divide_dict(idx_dict_list, batch_size):
    '''  '''
    total = len(idx_dict_list)
    if batch_size <= 0 or total <= batch_size:
        newid_batch = np.array(idx_dict_list)
        return [newid_batch]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            newid_batch = np.array(idx_dict_list[beg:beg + batch_size])
            batchs_list.append(newid_batch)

        beg = beg + batch_size
        diff = total - beg
        if diff > 0:
            if diff < batch_size/4:#
                newid_batch = batchs_list[-1]
                newid_batch = np.vstack((newid_batch, np.array(idx_dict_list[beg:])))
                batchs_list[-1] = newid_batch
            else:
                newid_batch = np.array(idx_dict_list[beg:])
                batchs_list.append(newid_batch)
        return batchs_list


def divide_array(idx_dict_array, batch_size):
    '''  '''
    total = idx_dict_array.shape[0]
    if batch_size <= 0 or total <= batch_size:
        newid_batch = idx_dict_array
        return [newid_batch]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            newid_batch = idx_dict_array[beg:beg + batch_size, :]
            batchs_list.append(newid_batch)

        beg = beg + batch_size
        diff = total - beg
        if diff > 0:
            if diff < batch_size/4:#
                newid_batch = batchs_list[-1]
                newid_batch = np.vstack((newid_batch, idx_dict_array[beg:, :]))
                batchs_list[-1] = newid_batch
            else:
                newid_batch = idx_dict_array[beg:, :]
                batchs_list.append(newid_batch)
        return batchs_list

