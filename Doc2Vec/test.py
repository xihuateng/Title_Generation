import os, sys
import json
train_data_path = sys.path[0] + "/news2016zh_train.json"
train_data_new = sys.path[0] + "/news2016zh_train500000.json"

i = 0
file = open(train_data_new, 'a+', encoding="utf-8")

for line in open(train_data_path,"r", encoding="utf-8"):
    a = json.loads(line)
    file.write(json.dumps(a, ensure_ascii=False)+'\n')
    i += 1
    if i == 500000:
        break
    if i % 1000 == 0:
        print(i)

file.close()
