#-*- encoding:utf-8 -*-

import os, sys
import json
import numpy as np
sys.path.append(sys.path[0]+'/textrank4zh/')
sys.path.append(sys.path[0]+'/bert4keras/')
from TextRank4Sentence import TextRank4Sentence
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
train_data_path = sys.path[0] + "/news2016zh_test.json"
title_extract_path = sys.path[0] + "/title_extract_bert.txt"

config_path = sys.path[0] + '/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = sys.path[0] + '/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = sys.path[0] + '/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

''' test
text = "中新网北京12月1日电(记者 张曦) 30日晚，高圆圆和赵又廷在京举行答谢宴，诸多明星现身捧场，其中包括张杰(微博)、谢娜(微博)夫妇、何炅(微博)、蔡康永(微博)、徐克、张凯丽、黄轩(微博)等。30日中午，有媒体曝光高圆圆和赵又廷现身台北桃园机场的照片，照片中两人小动作不断，尽显恩爱。事实上，夫妻俩此行是回女方老家北京举办答谢宴。群星捧场 谢娜张杰亮相当晚不到7点，两人十指紧扣率先抵达酒店。"
tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source = 'all_filters')

print('摘要：')
for item in tr4s.get_key_sentences(num=3):
    print(item.index, item.weight, item.sentence)
'''

def data_generator(N):
    # 数据生成器
    #n = 0
    for line in open(train_data_path,"r",encoding="utf-8"):
        #if n >= N:
        #    return
        a = json.loads(line)
        X = a['content']
        Y = a['title']
        yield X, Y
        #n += 1

def get_shorter_title():
    for item in tr4s.get_key_sentences(num=3,sentence_min_len = 5):
        if len(item['sentence']) < 50:
            return item['sentence']
    return "最新新闻报道"


tr4s = TextRank4Sentence()

title_extract = []

cnt = 0
for text, title in data_generator(10):
    tr4s.analyze(text=text, lower=True, source = 'no_filter', m = model, tz = tokenizer)
    if len(tr4s.get_key_sentences(num=1,sentence_min_len = 5)) < 1:
        title_extract.append('最新新闻报道')
    else:
        if len(tr4s.get_key_sentences(num=1,sentence_min_len = 5)[0]['sentence']) > 50:
            title_extract.append(get_shorter_title())
        else:
            title_extract.append(tr4s.get_key_sentences(num=1,sentence_min_len = 5)[0]['sentence'])

    cnt += 1
    if cnt % 500 == 0:
        print(cnt)
        f = open(title_extract_path,'a+',encoding='utf-8')
        for item in title_extract:
            f.write(item+'\n')
        f.close()
        title_extract = []
