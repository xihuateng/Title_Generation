#-*- encoding:utf-8 -*-

import os, sys
import json
import numpy as np
sys.path.append(sys.path[0]+'/textrank4zh/')
from TextRank4Sentence import TextRank4Sentence

train_data_path = sys.path[0] + "/news2016zh_train.json"
WORD_EMBEDDING = sys.path[0] + "/w2v.word"
#WORD_EMBEDDING = sys.path[0] + "/sgns.sogou.word"

