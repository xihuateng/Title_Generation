#-*- encoding:utf-8 -*-

import os, sys
import json
import numpy as np
sys.path.append(sys.path[0]+'/textrank4zh/')
from TextRank4Sentence import TextRank4Sentence

train_data_path = sys.path[0] + "/news2016zh_train.json"
WORD_EMBEDDING = sys.path[0] + "/sgns.sogou.word"

'''
n = 0
for line in open(train_data_path,"r",encoding="utf-8"):
    a = json.loads(line)
    X = a['content']
    Y = a['title']
    if n == 2:
        print(X)
        break
    n += 1
'''

embeddings_index = {}
f = open(WORD_EMBEDDING, encoding='utf8')

#取出word及其对应的embeddings，存入字典embeddings_index
cnt = 0
for line in f:
    if cnt == 0: #跳过第一行
        cnt += 1
        continue

    if cnt % 500 == 0:
        print(cnt)

    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = vec.tolist()
    cnt += 1
f.close()
print('Found %s word vectors.' % len(embeddings_index))
#print(embeddings_index)

'''
glove_embedding_matrix = np.zeros((NB_WORDS, EMBEDDING_DIM)) #申请0数组，
for word, i in word_index.items():
    if i < NB_WORDS+1: #+1 for 'unk' oov token
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            glove_embedding_matrix[i] = embedding_vector
        else:
            # 在embeddings索引中找不到的单词，将是unk的embeddings
            glove_embedding_matrix[i] = embeddings_index.get('unk')
print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0))
'''