import math
import os
import time
import random
import numpy as np

from datasetPath import fileUtil

def savePath(filename, path_list):
    with open(filename, 'w', encoding='utf-8') as fw:
        for line in path_list:
            fw.write(line.__str__() + '\n')


def getTriple(rel_triple): # , train_ent_id
    r_head = {}  # h:(r,t)

    r_tail = {}  # h:(r,t)
    tail_triples = []
    for (h, r, t) in rel_triple:
        r_head[h] = r_head.get(h, list()) + [(r, t)]

        r_tail[t] = r_tail.get(t, list()) + [(r, h)]
        tail_triples.append([t, r, h])

    return r_head, rel_triple, r_tail, tail_triples


def set_rPath_2(r_head, rel_triples, noRepeat=False):  

    path_list = []
    rpath_list = []
    for path in rel_triples: #(h, r, t)
        r1, t1 = path[1:]
        if t1 in r_head.keys():  # path[-1]=t
            for r2, t2 in r_head[t1]:  # (h, ..., r, t1) + (t1, r2, t2)
                if r1 == r2 and noRepeat:
                    continue
                else:
                    path_list.append(path + [r2, t2])  # t == hh
                    rpath_list.append((r1, r2))

    return rpath_list, path_list


def get_pathTrain(datasetPath):
    kgs_num_dict = fileUtil.load_dict(datasetPath + 'pre/kgs_num', read_kv='kv')
    kg_E = int(kgs_num_dict['KG_E'])  # KG_E

    rel_triple = fileUtil.load_triples_id(datasetPath + 'pre/rel_triples_id') 
    r_head, head_triple, r_tail, tail_triples = getTriple(rel_triple)

    head_rpath_list, head_path2_list = set_rPath_2(r_head, head_triple, noRepeat=False)
    print('head_path2_list:', len(head_path2_list))
    #tail
    tail_rpath_list, tail_path2_list = set_rPath_2(r_tail, tail_triples, noRepeat=True)
    print('tail_path2_list:', len(tail_path2_list))

    all_rpaht_list = head_rpath_list + tail_rpath_list
    rpath_dict = {}
    for r1, r2 in set(all_rpaht_list):
        rpath_dict[(r1, r2)] = 0
    for r1, r2 in all_rpaht_list:
        rpath_dict[(r1, r2)] += 1

    rpath_sort_all = sorted(rpath_dict.items(), key=lambda x:x[1], reverse=True)
    print('rpath_sort len:', len(rpath_sort_all))
    print('rpath_sort max:', rpath_sort_all[0])
    print('rpath_sort min:', rpath_sort_all[-1])

    savePath(datasetPath + 'pre/path/rpath_sort_all.txt', rpath_sort_all)
    print('rpath_sort_all:', len(rpath_sort_all))
    # 
    rpath_sort_dict = dict()
    for index, (path, num) in enumerate(rpath_sort_all):
        if num >= 50:
            rpath_sort_dict[path] = index
        else:
            break
    savePath(datasetPath + 'pre/rpath_sort_dict', list(rpath_sort_dict.items()))
    print('After 1 remove rpath>=50:', len(rpath_sort_dict))
    #path_path_triple = []
    path_neigh_dict = dict()
    path_triple_num = 0
    for eid in range(kg_E):
        path_neigh_dict[eid] = []
    for h, r1, t1, r2, t2 in head_path2_list + tail_path2_list:
        if (r1, r2) in rpath_sort_dict.keys():
            #path_path_triple.append((h, rpath_sort_dict[(r1, r2)], t2))
            rr_id = rpath_sort_dict[(r1, r2)]
            path_neigh_dict[h].append((t2, rr_id))
            path_triple_num += 1
    print('After 1 remove path_triple:', path_triple_num)

    ##############
    max_neighbors_num_all = len(max(path_neigh_dict.values(), key=lambda x: len(x)))
    print('Maximum number of neighbors:' + str(max_neighbors_num_all))  # max_neighbors_num = 235
    max_neighbors_num = 1000  # neighbors:9516
    path_triple_num = 0
    path_set = set()
    for h, trlist in path_neigh_dict.items():
        if len(trlist) > max_neighbors_num:
            trlist_sort = sorted(trlist, key=lambda x: x[1], reverse=False)
            trlist_new = trlist_sort[:max_neighbors_num]
        else:
            trlist_new = trlist
        path_neigh_dict[h] = trlist_new
        for t,r in trlist_new:
            path_set.add(r)
        path_triple_num += len(path_neigh_dict[h])
    savePath(datasetPath + 'pre/path_neigh_dict', list(path_neigh_dict.items()))
    print('After 2 remove path_triple:' + str(path_triple_num))
    print('After 2 remove rpath_sort_all:' + str(len(path_set)))


if __name__ == '__main__':
    print("start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

    #datasetPath = '../../0908datasets/DBP15K/zh_en(dbp15)/' # fr_en(dbp15) ja_en(dbp15), zh_en(dbp15)
    datasetPath = '../../0908datasets/WN31/EN_FR_15K_V2/' # EN_DE_15K_V1、EN_FR_15K_V1
    #datasetPath = '../../0908datasets/DWY100K/dbp_wd/' # DWY100K/dbp_wd, DWY100K/dbp_yg

    if not os.path.exists(datasetPath+'pre/path/'):
        os.makedirs(datasetPath+'pre/path/')

    # seed = 72
    # ordered = True
    # print(datasetPath)
    # random.seed(seed)
    # np.random.seed(seed)

    get_pathTrain(datasetPath)

