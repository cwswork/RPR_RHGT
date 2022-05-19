import copy
import os
import json
import torch
import numpy as np

from align import pre_relScore
from autil import fileUtil


class load_KGs_data(object):
    def __init__(self, myconfig):
        self.myconfig = myconfig
        # Load Datasets
        kgs_num_dict = fileUtil.load_dict(myconfig.datasetPath + 'kgs_num', read_kv='kv')
        self.kg_E = int(kgs_num_dict['KG_E'])  # KG_E
        self.kg_R = int(kgs_num_dict['KG_R'])

        ## Relation triple
        self.rel_triples = fileUtil.load_triples_id(myconfig.datasetPath + 'rel_triples_id')
        ## entity embedding
        if myconfig.embed_type == 1:
            ename_embed = fileUtil.loadpickle(myconfig.datasetPath + 'entity_embedding.out')
        else: # myconfig.embed_type = 2
            with open(file=myconfig.datasetPath + 'vectorList.json', mode='r', encoding='utf-8') as f:
                embedding_list = json.load(f)
                print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
                ename_embed = np.array(embedding_list)
        self.ename_embed = torch.FloatTensor(ename_embed)

        ## train、Valid、Test  # np.array
        if '100' in myconfig.datasetPath:
            train_links_id = fileUtil.get_links_ids(myconfig.tt_path + 'train_links_id')
            valid_links_id = []
            test_links_id = fileUtil.get_links_ids(myconfig.tt_path + 'test_links_id')
        else:
            train_links_id = fileUtil.get_links_ids(myconfig.tt_path + 'train_links_id')
            valid_links_id = fileUtil.get_links_ids(myconfig.tt_path + 'valid_links_id')
            test_links_id = fileUtil.get_links_ids(myconfig.tt_path + 'test_links_id')

        self.train_links = train_links_id
        self.valid_links = valid_links_id
        self.test_links = test_links_id

        print('out_temp:', myconfig.out_temp)
        # rel
        self.pre_relation(myconfig)
        # path
        self.pre_path(myconfig)


    def pre_relation(self, myconfig):
        myconfig.myprint("\n=== pre_relation ==")
        ent_neigh_dict = dict()

        # self
        rel_self = self.kg_R
        self.kg_R += 1
        for eid in range(self.kg_E):
            ent_neigh_dict[eid] = [(eid, rel_self)]
        rel_triple_num = 0
        for (h, r, t) in self.rel_triples:  # 无方向
            ent_neigh_dict[h].append((t, r))
            rel_triple_num += 1
            if h != t:
                ent_neigh_dict[t].append((h, r))
                rel_triple_num += 1
        self.ent_neigh_dict = ent_neigh_dict
        myconfig.myprint('ent_neigh_dict:' + str(len(self.ent_neigh_dict)))
        myconfig.myprint('rel_triple_num:' + str(rel_triple_num))


    def pre_path(self, myconfig):
        myconfig.myprint("\n=== pre_path ==")
        # path_neigh_dict: Path and its associated head and tail entities
        path_neigh_dict = dict()
        with open(myconfig.datasetPath + 'path_neigh_dict', 'r', encoding='utf-8') as fr:
            for line in fr:
                entid, rtlist = eval(line[:-1])
                path_neigh_dict[entid] = rtlist

        # rpath_sort_dict: Paths and their frequency numbers
        pathid2rr = dict()
        with open(myconfig.datasetPath + 'rpath_sort_dict', 'r', encoding='utf-8') as fr:
            for line in fr:
                rpath, pathid = eval(line[:-1])
                pathid2rr[pathid] = rpath

        # 2、path_pair_dict
        path_len = len(pathid2rr)
        path_pair_dict, temp_path_list, temp_notpath_list = pre_relScore.get_align_path(self.ename_embed, self.train_links, path_neigh_dict,
                                                                                self.kg_E, path_len)
        myconfig.myprint("Number of path_pair_dict:" + str(len(path_pair_dict)))
        fileUtil.save_list2txt(myconfig.out_temp + 'temp_path_list.txt', temp_path_list)
        fileUtil.save_list2txt(myconfig.out_temp + 'temp_notpath_list.txt', temp_notpath_list)

        # 5、pathid
        kg1_id2rel = fileUtil.load_ids2dict(myconfig.datasetPath + 'kg1_rel_dict', read_kv='kv')  # name:id
        kg2_id2rel = fileUtil.load_ids2dict(myconfig.datasetPath + 'kg2_rel_dict', read_kv='kv')
        kg1_id2rel.update(kg2_id2rel)

        oldpath2newpath = dict()
        path_save_list = []
        for path_newid, (path1, path2) in enumerate(path_pair_dict.items()):
            oldpath2newpath[path1] = oldpath2newpath[path2] = path_newid
            path1_r1, path1_r2 = pathid2rr[path1]
            path2_r1, path2_r2 = pathid2rr[path2]
            path_save_list.append((path_newid, (path1, pathid2rr[path1]), (path2, pathid2rr[path2]),
                                   (path1, kg1_id2rel[path1_r1], kg1_id2rel[path1_r2]),
                                   (path2, kg1_id2rel[path2_r1], kg1_id2rel[path2_r2])))
        fileUtil.save_list2txt(myconfig.out_temp + 'path_save_list.txt', path_save_list)

        path_triple_num = 0
        path_triple_old_num = 0
        # self
        path_self_id = path_newid + 1
        for h, trlist in path_neigh_dict.items():
            path_triple_old_num += len(trlist)
            trlist_new = [(t, oldpath2newpath[path_oldid]) for (t, path_oldid) in trlist if path_oldid in oldpath2newpath.keys()]
            if len(trlist_new) <= 1:
                trlist_new.append((h, path_self_id))
            path_neigh_dict[h] = trlist_new
            path_triple_num += len(trlist_new)

        self.kg_path = path_self_id + 1
        self.path_neigh_dict = path_neigh_dict
        fileUtil.save_list2txt(myconfig.out_temp + 'path_neigh_dict', list(path_neigh_dict.items()))
        myconfig.myprint("Number of path_newid:" + str(self.kg_path))
        myconfig.myprint("Number of all path_triple:" + str(path_triple_old_num))
        myconfig.myprint("Number of path_triple by aligned:" + str(path_triple_num))
