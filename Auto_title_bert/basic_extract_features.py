#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征
import sys
sys.path.append(sys.path[0]+'/bert4keras/')
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
import numpy as np

config_path = sys.path[0] + '/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = sys.path[0] + '/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = sys.path[0] + '/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'今天天气很好。')
token_ids, segment_ids = to_array([token_ids], [segment_ids])

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

print('\n ===== predicting =====\n')
a = model.predict([token_ids, segment_ids])

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'星期一可能要下雨，大家注意一下。')
token_ids, segment_ids = to_array([token_ids], [segment_ids])
b = model.predict([token_ids, segment_ids])
print(cos_sim(a[0][0], b[0][0]))

token_ids, segment_ids = tokenizer.encode(u'新华社的报道出现了偏差。')
token_ids, segment_ids = to_array([token_ids], [segment_ids])
c = model.predict([token_ids, segment_ids])
print(cos_sim(a[0][0], c[0][0]))
"""
输出：
[[[-0.63251007  0.2030236   0.07936534 ...  0.49122632 -0.20493352
    0.2575253 ]
  [-0.7588351   0.09651865  1.0718756  ... -0.6109694   0.04312154
    0.03881441]
  [ 0.5477043  -0.792117    0.44435206 ...  0.42449304  0.41105673
    0.08222899]
  [-0.2924238   0.6052722   0.49968526 ...  0.8604137  -0.6533166
    0.5369075 ]
  [-0.7473459   0.49431565  0.7185162  ...  0.3848612  -0.74090636
    0.39056838]
  [-0.8741375  -0.21650358  1.338839   ...  0.5816864  -0.4373226
    0.56181806]]]
"""
'''
print('\n ===== reloading and predicting =====\n')
model.save('test.model')
del model
model = keras.models.load_model('test.model')
print(model.predict([token_ids, segment_ids]))
'''