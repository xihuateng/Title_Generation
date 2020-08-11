#-*- encoding:utf-8 -*-

import os, sys
import json
import numpy as np
sys.path.append(sys.path[0]+'/textrank4zh/')
from TextRank4Sentence import TextRank4Sentence

train_data_path = sys.path[0] + "/news2016zh_valid.json"
title_extract_w2v_path = sys.path[0] + "/title_extract_w2v.txt"


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

    tr4s.get_key_sentences(num=2,sentence_min_len = 5)[0]['sentence']


tr4s = TextRank4Sentence()

title_extract = []

cnt = 0
for text, title in data_generator(10):

    tr4s.analyze(text=text, lower=True, source = 'no_filter')

    if len(tr4s.get_key_sentences(num=1,sentence_min_len = 5)) < 1:
        title_extract.append('最新新闻报道')
    else:
        if len(tr4s.get_key_sentences(num=1,sentence_min_len = 5)[0]['sentence']) > 50:
            title_extract.append(get_shorter_title())
        else:
            title_extract.append(tr4s.get_key_sentences(num=1,sentence_min_len = 5)[0]['sentence'])

    cnt += 1
    if cnt % 500 == 0:
        print(f"{cnt} titles extracted")
        f = open(title_extract_w2v_path,'a+',encoding='utf-8')
        for item in title_extract:
            f.write(item+'\n')
        f.close()
        title_extract = []
