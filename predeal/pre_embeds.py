import string
import time
import numpy as np
import pandas as pd


##  name embedding ##############################
def get_word_embed(entity_name_list):

    namesFrame = pd.DataFrame(entity_name_list)  
    names_len = len(namesFrame)
    print('namesFrame:', len(namesFrame))
    namesFrame.iloc[:, 1] = namesFrame.iloc[:, 1].str.replace(r'[{}]+'.format(string.punctuation), ' ').str.split(' ')  # 第3列改为list

    entity_name_new = []
    name_words_list = [] 
    for id, l in namesFrame.iloc[:].values:
        entity_name_new.append((id, ' '.join(l)))
        name_words_list += l
    name_words_list = list(set(name_words_list))  
    print('names_word:', len(name_words_list)) 

    # embedding
    word_embed_dict, unlisted_words = get_name_word_embed(name_words_list)  
    print('names word embed:', len(word_embed_dict))
    unlisted_words_embed = generate_word2vec_by_character_embedding(unlisted_words) 
    print('names char embed:', len(unlisted_words_embed))  
    word_embed_dict.update(unlisted_words_embed)  
    print('word_embed_dict(sum of up):', len(word_embed_dict)) 

    names_word_em = np.zeros([len(name_words_list) + 1, 300])  
    names_word_em = np.stack(names_word_em, axis=0).astype(np.float64) 
    count = 0
    for word_id in range(len(name_words_list)):
        if name_words_list[word_id] in word_embed_dict:
            names_word_em[word_id] = word_embed_dict[name_words_list[word_id]]
            count += 1
    print('count:', count) 

    names_words_se = pd.Series(name_words_list)
    names_words_se = pd.Series(names_words_se.index, names_words_se.values)

    def lookup_and_padding(x):
        default_length = 4
        ids = list(names_words_se.loc[x].values) + [names_words_se.iloc[-1], ] * default_length
        return ids[:default_length]

    namesFrame.iloc[:, 1] = namesFrame.iloc[:, 1].apply(lookup_and_padding)

    # entity-desc-embedding dataframe
    un_logged_id = len(names_words_se)
    e_desc_input = pd.DataFrame(np.repeat([[un_logged_id, ] * 4], names_len, axis=0),
                                range(names_len))
    e_desc_input.iloc[namesFrame.iloc[:, 0].values] = np.stack(namesFrame.iloc[:, 1].values)

    entity_embeds = names_word_em[e_desc_input.values]
    entity_embeds = np.sum(entity_embeds, axis=1)
    return entity_name_new, entity_embeds


def get_entity_embed(ent_id2value_list):
    '''    '''
    print('\nget entity embedding')
    start = time.time()

    new_ent_list = list() 
    for (e_id, e_name) in ent_id2value_list:
        e_name = e_name.split('/')[-1]
        new_ent_list.append((e_id, e_name))

    #aa = np.array(new_ent_list)
    clean_entity_list, name_embeds = get_word_embed(new_ent_list) 
    print('entity_embed_mat:', len(name_embeds))

    print('generating costs time: {:.4f}s'.format(time.time() - start))
    return clean_entity_list, name_embeds


def get_name_word_embed(names_words):
    word_embed_file = './../../datasets1012/wiki-news-300d-1M.vec'

    listed_words_dict = {}

    print('load word embedding')
    with open(word_embed_file, 'r', encoding='utf-8') as f:
        w = f.readlines()
        w = pd.Series(w[1:])  # SERICES:(999994,)
    begin, eachnum = 0, 200000
    allword_sum = len(w)
    while begin < allword_sum:
        end = begin + eachnum if begin + eachnum < allword_sum else allword_sum
        we = w[begin:end].str.split(' ')
        word = we.apply(lambda x: x[0])
        w_em = we.apply(lambda x: x[1:])
        we_dict = dict(zip(word, w_em))
        for v in names_words:
            if v in we_dict:
                listed_words_dict[v] = we_dict[v] 
        begin = end
        print(end, end=',')

    unlisted_words = []
    for w in names_words:
        if w not in listed_words_dict:
            unlisted_words.append(w) 

    return listed_words_dict, unlisted_words 


def generate_word2vec_by_character_embedding(word_list, vector_dimension=300):
    ''' char embedding '''
    character_vectors = {}
    alphabet = ''
    ch_num = {}  # 
    for word in word_list:
        for ch in word:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n
    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            alphabet += ch_num[i][0]

    char_sequences = [list(word) for word in word_list]
    model = Word2Vec(char_sequences, size=vector_dimension, window=5, min_count=1)
    for ch in alphabet:
        assert ch in model
        character_vectors[ch] = model[ch]

    get_word2vec = {}
    for word in word_list:
        vec = np.zeros(vector_dimension, dtype=np.float32)
        for ch in word:
            if ch in alphabet:
                vec += character_vectors[ch]
        if len(word) != 0:
            get_word2vec[word] = vec / len(word)
        # else:
        #     print(word)
    return get_word2vec


