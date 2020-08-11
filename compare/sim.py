#-*- encoding:utf-8 -*-

import os, sys
import numpy as np
from bert_serving.client import BertClient


title_path = sys.path[0] + "/title.txt"
title_extract_path = sys.path[0] + "/title_extract.txt"
title_gen_path = sys.path[0] + "/title_gen.txt"
title_extract_w2v_path = sys.path[0] + "/title_extract_w2v.txt"

title_vec = sys.path[0] + "/title_vec.npy"
title_extract_vec = sys.path[0] + "/title_extract_vec.npy"
title_gen_vec = sys.path[0] + "/title_gen_vec.npy"
title_extract_w2v_vec = sys.path[0] + "/title_extract_w2v.npy"



def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim * 100


bc = BertClient()
cnt = 0
title_v = []
for line in open(title_extract_w2v_path,"r",encoding="utf-8"):
    a = bc.encode([line.rstrip('\n')])
    title_v.append(a)
    cnt += 1
    if cnt % 100 == 0:
        print(cnt)
    
title_v = np.array(title_v)
np.save(title_extract_w2v_vec, title_v)


#-----------Cal Sim-----------
title_vec = np.load(title_vec)
title_vec = title_vec.tolist()
title_extract_w2v_vec = np.load(title_extract_w2v_vec)
title_extract_w2v_vec = title_extract_w2v_vec.tolist()

sim = 0.0
l = len(title_vec)
for i in range(l):
    s_i = cos_sim(title_vec[i], title_extract_w2v_vec[i])
    sim += s_i
    if i % 500 == 0:
        print(i)
sim /= l

print(f"The similarity of title and title_extract_w2v: {sim}%")

'''
title_vec = np.load(title_vec)
title_vec = title_vec.tolist()
title_extract_vec = np.load(title_extract_vec)
title_extract_vec = title_extract_vec.tolist()

sim = 0.0
l = len(title_vec)
for i in range(l):
    s_i = cos_sim(title_vec[i], title_extract_vec[i])
    sim += s_i
    if i % 500 == 0:
        print(i)
sim /= l

print(f"The similarity of title and title_extract: {sim}%")
#The similarity of title and title_extract: 93.66692190107987%

title_vec = np.load(title_vec)
title_vec = title_vec.tolist()
title_gen_vec = np.load(title_gen_vec)
title_gen_vec = title_gen_vec.tolist()

sim = 0.0
l = len(title_vec)
for i in range(l):
    s_i = cos_sim(title_vec[i], title_gen_vec[i])
    sim += s_i
    if i % 500 == 0:
        print(i)
sim /= l

print(f"The similarity of title and title_gen: {sim}%")
#The similarity of title and title_gen: 93.95291036456518%
'''