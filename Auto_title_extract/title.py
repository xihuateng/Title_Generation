#-*- encoding:utf-8 -*-

import os, sys
import json
train_data_path = sys.path[0] + "/news2016zh_valid.json"

cnt = 0
title = []
for line in open(train_data_path,"r",encoding="utf-8"):
    a = json.loads(line)
    Y = a['title']
    title.append(Y)
    cnt += 1

    if cnt % 5 == 0:
        print(cnt)
        f = open(sys.path[0] + "/title.txt",'a+',encoding='utf-8')
        for item in title:
            f.write(item+'\n')
        f.close()
        title = []

