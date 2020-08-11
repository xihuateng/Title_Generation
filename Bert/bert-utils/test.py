#-*- encoding:utf-8 -*-

import os, sys
sys.path.append(sys.path[0])

from extract_feature import BertVector
bv = BertVector()
bv.encode(['今天天气不错'])
