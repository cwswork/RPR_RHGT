import os
import time
import random
import numpy as np
from predeal import fileUtil
from predeal import pre_embeds



def set_relation2id_byID(datasetPath):
    # entity
    kg1_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_1', read_kv='vk')  # (name:eid) - id:name
    kg2_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_2', read_kv='vk')  # en
    fileUtil.save_dict2txt(datasetPath + 'pre/kg1_ent_dict', kg1_ent2id_dict, save_kv='vk')
    fileUtil.save_dict2txt(datasetPath + 'pre/kg2_ent_dict', kg2_ent2id_dict, save_kv='vk')

    print("Num of KG1 entitys:", len(kg1_ent2id_dict))
    print("Num of KG2 entitys:", len(kg2_ent2id_dict))

    # relation ######
    kg1_rel_triples_id, kg1_entities_id, kg1_relations_id = fileUtil.read_relation_triples(datasetPath + 'triples_1')
    kg2_rel_triples_id, kg2_entities_id, kg2_relations_id = fileUtil.read_relation_triples(datasetPath + 'triples_2')
    print("Num of KG1 relations:", len(kg1_relations_id))
    print("Num of KG2 relations:", len(kg2_relations_id))
    print("Num of KG1 relation triples:", len(kg1_rel_triples_id))
    print("Num of KG2 relation triples:", len(kg2_rel_triples_id))

    rel_triples_id = list(kg1_rel_triples_id) + list(kg2_rel_triples_id)
    print('rel_triples_id:', len(rel_triples_id))
    rel_triples_id = list(set(rel_triples_id))
    print('rel_triples_id set :', len(rel_triples_id))
    fileUtil.save_triple2txt(datasetPath + 'pre/rel_triples_id', rel_triples_id)
    # save
    KG_E = len(kg1_ent2id_dict) + len(kg2_ent2id_dict)
    KG_R = len(kg1_relations_id) + len(kg2_relations_id)
    print()
    print("Num of KGs entitys:", KG_E)
    print("Num of KGs relations:", KG_R)
    print("Num of KGs relation triples:", len(rel_triples_id))

    with open(datasetPath + 'pre/kgs_num', 'w') as ff:
        ff.write('KG_E:' + str(KG_E) + '\n')
        ff.write('KG_R:' + str(KG_R) + '\n')


##embedding###############
def get_entity_embed(datasetPath):
    # entity embedding  # ja,zh
    kg1_ent_list = fileUtil.load_ids2list(datasetPath + 'ent_ids_1')
    kg2_ent_list = fileUtil.load_ids2list(datasetPath + 'ent_ids_2')
    kg_ent_list = kg1_ent_list + kg2_ent_list

    kg_ent_list = sorted(kg_ent_list, key= lambda x:x[0], reverse=False)
    a = kg_ent_list[-1]
    new_entity_name_list, entity_embedding = pre_embeds.get_entity_embed(kg_ent_list)
    fileUtil.save_list2txt(datasetPath + 'pre/ent_dict_replace_name.txt', new_entity_name_list)
    fileUtil.savepickle(datasetPath + 'pre/entity_embedding.out', entity_embedding) 


###  ###
def set_links_file(datasetPath, dataset_division):
    if '100' in datasetPath:
        ill_file = 'ref_ent_ids'
    else:
        ill_file = 'ill_ent_ids'
    ILL = fileUtil.load_triples_id(datasetPath + ill_file)
    kg1_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_1', read_kv='kv')
    kg2_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_2', read_kv='kv')
    IIL_ids = []
    for e1, e2 in ILL:
        IIL_ids.append((kg1_ent2id_dict[e1], kg2_ent2id_dict[e2]))

    ILL_len = len(IIL_ids)  # illL=15000
    np.random.shuffle(IIL_ids)
    train_links = np.array(IIL_ids[:ILL_len // 10 * 3])  # 30%
    #valid_links = np.array(IIL_ids[len(train_links):ILL_len // 10 * 3])  # 10%
    test_links = IIL_ids[ILL_len // 10 * 3:]  # 70%

    print('save files...train_links...')
    fileUtil.save_list2txt(datasetPath + dataset_division + 'train_links', train_links)
    #fileUtil.save_list2txt(datasetPath + dataset_division + 'valid_links', valid_links)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'test_links', test_links)


def set_links_ID(datasetPath, dataset_division):
    kg1_entity2index = fileUtil.load_ids2dict(datasetPath + 'ent_ids_1', read_kv='vk')
    kg2_entity2index = fileUtil.load_ids2dict(datasetPath + 'ent_ids_2', read_kv='vk')

    train_links_ID = fileUtil.get_links2ids(datasetPath + dataset_division + 'train_links',
                                              kg1_entity2index, kg2_entity2index)
    test_links_ID = fileUtil.get_links2ids(datasetPath + dataset_division + 'test_links',
                                             kg1_entity2index, kg2_entity2index)
    # valid_links_ID = fileUtil.get_links2ids(datasetPath + dataset_division + 'valid_links',
    #                                           kg1_entity2index, kg2_entity2index)
    print('save files...train_links...')
    fileUtil.save_list2txt(datasetPath + dataset_division + 'train_links_id', train_links_ID)
    #fileUtil.save_list2txt(datasetPath + dataset_division + 'valid_links_id', valid_links_ID)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'test_links_id', test_links_ID)


if __name__ == '__main__':
    print("start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

    # EN_DE_15K_V1, EN_FR_15K_V1, DBP15K/fr_en(dbp15) ja_en(dbp15), zh_en(dbp15)
    #datasetPath = '../../0908datasets/DBP15K/zh_en(dbp15)/' # fr_en(dbp15) ja_en(dbp15), zh_en(dbp15)
    datasetPath = '../../0908datasets/WN31/EN_FR_15K_V2/' # EN_DE_15K_V1、EN_FR_15K_V1
    #datasetPath = '../../0908datasets/DWY100K/dbp_wd/' # DWY100K/dbp_wd, DWY100K/dbp_yg

    seed = 72  # seed =72, 3, 26, 728, 20
    print(datasetPath)
    random.seed(seed)
    np.random.seed(seed)
	
	set_relation2id_byID(datasetPath)
	get_entity_embed(datasetPath)
    # 
    dataset_division = '721_5fold/1/'  # pre/30/
    if not os.path.exists(datasetPath + dataset_division):
        print('dir not exists：' + dataset_division)
        os.makedirs(datasetPath + dataset_division)
	
    # set_links_file(datasetPath, dataset_division)
    set_links_ID(datasetPath, dataset_division)


    print("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
