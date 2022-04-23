import copy
import json
import random

train_path = "../datasets/split_data/train.json"
dic_path = "../datasets/split_data/features/dic/train_dic.json"
output_path = "../datasets/split_data/enhanced_train.json"

dic_list = json.load(open(dic_path, "r"))
data = json.load(open(train_path, "r"))

# 深拷贝数据防止变更
original_data = copy.deepcopy(data)

for i in data:
    word_list = i[1]
    for idx, word in enumerate(word_list):
        repalce_word = random.choice(dic_list)
        word_list[idx] = repalce_word
        i[0] = i[0].replace(word, repalce_word)

original_data.extend(data)
print("original data has {} instances.".format(len(data)))
print("output data has {} instances and saved in {}.".format(len(original_data), output_path))
json.dump(original_data, open(output_path, "w"), ensure_ascii=False)
