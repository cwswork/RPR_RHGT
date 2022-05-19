# RPR-RHGT

Source code and datasets for 2022 paper: 
[***Entity Alignment with Reliable Path Reasoning and Relation-aware Heterogeneous Graph Transformer***]

## Datasets

> Please first download the main datasets [here](https://www.jianguoyun.com/p/DY8iIAsQ2t_lCBjK3oUEIAA) 
, path datasets [here](https://www.jianguoyun.com/p/DWzhBksQ2t_lCBjon78EIAA)
and extract them into `datasets/` directory.

Initial datasets WN31-15K and DBP-15K are from [OpenEA](https://github:com/nju-websoft/OpenEA) and [JAPE](https://github.com/nju-websoft/JAPE).

Initial datasets DWY100K is from  [BootEA](https://github.com/nju-websoft/BootEA).

Take the dataset EN_DE(V1) as an example, the folder "pre " of main datasets contains:
* kg1_ent_dict: ids for entities in source KG;
* kg2_ent_dict: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* rel_triples_id: relation triples encoded by ids;
* kgs_num: statistics of the number of entities, relations, attributes, and attribute values;
* entity_embedding.out: the input entity name feature matrix initialized by word vectors;

The folder "pre " of path datasets contains:
* path_neigh_dict: Path and its associated head and tail entities;
* rpath_sort_dict: Paths and their frequency numbers;

## Environment

* Python>=3.7
* pytorch>=1.7.0
* tensorboardX>=2.1.0
* Numpy
* json


## Running

To run RPR-RHGT model on WN31-15K and DBP-15K, use the following script:
```
python3 align/exc_plan.py
```
To run RPR-RHGT model DWY100K, use the following script:
```
python3 align100K/exc_plan100K.py
```

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.
> If you have any difficulty or question in running code and reproducing expriment results, please email to cwswork@qq.com.

## Citation

If you use this model or code, please cite it as follows:

*Weishan Cai, Wenjun Ma, Jieyu Zhan, and Yuncheng Jiang*, 
“Entity Alignment with Reliable Path Reasoning and Relation-aware Heterogeneous Graph Transformer”. In [Proceedings of the 31st International Joint Conference on Artificial Intelligence and the 25th European Conference on Artificial Intelligence (IJCAI-ECAI 2022)](https://ijcai-22.org/main-track-accepted-papers/)
