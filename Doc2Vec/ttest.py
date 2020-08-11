#-*- encoding:utf-8 -*-
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

    return cos

model = Doc2Vec.load(sys.path[0]+'/model/modeltest',mmap='r')
#推测文本的向量
#model.random.seed(0)
Vector1 = model.infer_vector(['新华社','报道','出现','偏差'],steps=500,alpha=0.025)
Vector2 = model.infer_vector(['新华社','的', '报道','出现','错误'],steps=500,alpha=0.025)
Vector3 = model.infer_vector(['今天','的','天气','非常','好'],steps=500,alpha=0.025)
print(similarity(Vector1,Vector2))
print(similarity(Vector1,Vector3))