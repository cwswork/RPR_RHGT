# RPR_RHGT
Source code and datasets for 2022 paper: [***Entity Alignment with Reliable Path Reasoning and Relation-aware Heterogeneous Graph Transformer***]

## Datasets

> Please first download the datasets [here](https://www.jianguoyun.com/p/DegoGgMQ4vHdCRin8oUE) and extract them into `datasets/` directory.

Initial datasets WN31-15K and DBP-15K are from [OpenEA](https://github:com/nju-websoft/OpenEA) and [JAPE](https://github.com/nju-websoft/JAPE).

Initial datasets DWY100K is from  [BootEA](https://github.com/nju-websoft/BootEA) and [MultiKE](https://github.com/nju-websoft/MultiKE).

Take the dataset EN_DE(V1) as an example, the folder "pre " contains:
* kg1_ent_dict: ids for entities in source KG;
* kg2_ent_dict: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* rel_triples_id: relation triples encoded by ids;
* kgs_num: statistics of the number of entities, relations, attributes, and attribute values;
* entity_embedding.out: the input attribute value feature matrix initialized by word vectors;


## Environment

* Python>=3.7
* pytorch>=1.7.0
* tensorboardX>=2.1.0
* Numpy
* json
