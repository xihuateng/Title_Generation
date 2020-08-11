#-*- encoding:utf-8 -*-
import os, sys
sys.path.append(sys.path[0]+'/textrank4zh/')
from TextRank4Sentence import TextRank4Sentence
#from TextRank4Keyword import TextRank4Keyword
import json
train_data_path = sys.path[0] + "/news2016zh_train.json"

import util
import jieba
import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from Segmentation import *
from gensim.models.callbacks import CallbackAny2Vec

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def get_dataset():
    #获取分句后的文本
    stop_words_file = get_default_stop_words_file()
    seg = Segmentation( stop_words_file=stop_words_file,
                        allow_speech_tags=util.allow_speech_tags,
                        delimiters=util.sentence_delimiters)
    text = []
    cnt = 0 
    for line in open(train_data_path,"r",encoding="utf-8"):
        a = json.loads(line)
        result = seg.segment(text=a['content'], lower=False)
        text = text + result.sentences
        cnt += 1
        if cnt % 1000 == 0:
            print(f'{cnt} content finished.')
    print(f'{len(text)} sentences')
    return text

def get_cutsentences(text):
    #句子分词，去停用词
    stop_words_file = get_default_stop_words_file()
    stop_list = [line[:-1] for line in open(stop_words_file,'r')]
    result = []
    cnt = 0
    for s in text:
        s_cut = jieba.cut(s)
        s_split = ' '.join(s_cut).split()
        s_result = [word for word in s_split if word not in stop_list]
        result.append(' '.join(s_result))
        cnt += 1
        if cnt % 1000 == 0:
            print(f'{cnt} sentences finished.')
    return result

TaggededDocument = gensim.models.doc2vec.TaggedDocument
def X_train(result):
    #调整Train格式
    x_train = []
    for i, text in enumerate(result):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list,tags=[i])
        x_train.append(document)
    return x_train

def train(x_train, size=300): ##size训练出的句子向量的维度
    epoch_logger = EpochLogger()
    model = Doc2Vec(x_train, min_count=1, window=5, vector_size=size, sample=1e-3, negative=5, workers=4,callbacks=[epoch_logger])
    model.train(x_train, total_examples=model.corpus_count, epochs=50)
    model.save(sys.path[0]+'/model/d2v') ##模型保存的位置
    return model


def ceshi():
    model = Doc2Vec.load(sys.path[0]+'/model/d2v')
    ##此处需要读入你所需要进行提取出句子向量的文本   此处代码需要自己稍加修改一下
    str1 = '2018年03月23日晚上大概十一点多钟我和张三骑着摩托车从住处出门想看看有什么能吃的东西.'
    ##你需要进行得到句子向量的文本，如果是分好词的，则不需要再调用结巴分词
    test_text = ' '.join(jieba.cut(str1)).split()

    inferred_vector_dm = model.infer_vector(test_text,steps=500,alpha=0.025) ##得到文本的向量
    
    Vector1 = model.infer_vector(['新华社','报道','出现','偏差'],steps=500,alpha=0.025)
    Vector2 = model.infer_vector(['报纸','的', '新闻','有','错误'],steps=500,alpha=0.025)
    Vector3 = model.infer_vector(['今天','的','天气','非常','好'],steps=500,alpha=0.025)
    print(similarity(Vector1,Vector2))
    print(similarity(Vector2,Vector3))

    return inferred_vector_dm

def similarity(a_vect, b_vect):
    #计算两个向量余弦值
    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = None
    for a, b in zip(a_vect, b_vect):
        dot_val += a*b
        a_norm += a**2
        b_norm += b**2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm*b_norm)**0.5)

    return cos*0.5 + 0.5


if __name__ == '__main__':
    
    text = get_dataset()
    print('Get dataset finished.')
    result = get_cutsentences(text)
    print('Cut sentences finished.')
    x_train = X_train(result)
    print('x train finished')
    model_dm = train(x_train)
    print('model train finished')
    
    doc_2_vec = ceshi()
    #print (doc_2_vec)
    #print(doc_2_vec.shape)

    
