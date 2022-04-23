"""
只保留 tarin 中的 phase
"""
import json

train_path = "../../train.json"
data = json.load(open(train_path, "r"))

words = []
for i in data:
    words.extend(i[1])
    
json.dump(words, open("train_dic.json", 'w'), ensure_ascii=False)