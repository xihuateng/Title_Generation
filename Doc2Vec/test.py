import os, sys
sys.path.append(sys.path[0]+'/textrank4zh/')
from TextRank4Sentence import TextRank4Sentence
#from TextRank4Keyword import TextRank4Keyword
import json
train_data_path = sys.path[0] + "/news2016zh_s.json"

import util
import jieba
import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from Segmentation import *


stop_words_file = get_default_stop_words_file()

#-----文本分句
seg = Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=util.allow_speech_tags,
                                delimiters=util.sentence_delimiters)

text = []
for line in open(train_data_path,"r",encoding="utf-8"):
    a = json.loads(line)
    result = seg.segment(text=a['content'], lower=False)
    text = text + result.sentences

print(len(text))


#-----分词并删除停用词

stop_list = [line[:-1] for line in open(stop_words_file,'r')]
result = []
for s in text:
    s_cut = jieba.cut(s)
    s_split = ' '.join(s_cut).split()
    s_result = [word for word in s_split if word not in stop_list]
    result.append(' '.join(s_result))

#-----训练格式
TaggededDocument = gensim.models.doc2vec.TaggedDocument
x_train = []
for i, text in enumerate(result):
    word_list = text.split(' ')
    l = len(word_list)
    word_list[l-1] = word_list[l-1].strip()
    document = TaggededDocument(word_list,tags=[i])
    x_train.append(document)
print(x_train)

model = Doc2Vec(x_train, min_count=1, window=5, vector_size=300, sample=1e-3, negative=5, workers=4)
model.train(x_train, total_examples=model.corpus_count, epochs=50)
model.save(sys.path[0]+'/model/modeltest') ##模型保存的位置

