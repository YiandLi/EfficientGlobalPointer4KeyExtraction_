"""
对下载好的w2v进行处理，word只保留中文
输出字典的json文件
"""

import json
import re

# 下载好的 w2v 数据，每行一个 word+300维向量
w2v_path = 'data/sgns.financial.word'
# 输出字典 json 地址
output_path = "data/word2vec_dic.json"

with open(w2v_path, 'r', encoding='utf-8') as f:
    instance_list = f.readlines()

incomplete_num = 0  # 记录向量确实的个数（不够 300 维度）
duplicate_num = 0  # 记录重复word的个数

w2v_dict = {}
for index, i in enumerate(instance_list):
    if index == 0:
        continue
    # 只保留中，英："[^\u4e00-\u9fa5^a-z^A-Z]"
    # 保留中，英，数字："[^\u4e00-\u9fa5^a-z^A-Z^0-9]"
    # 只保留中文： "[^\u4e00-\u9fa5]"
    real_word = re.sub(r"[^\u4e00-\u9fa5]", "", i.split(' ')[0])
    real_emb = i.strip().split(' ')[1:]
    
    # 如果没有中文部分，直接跳过
    if real_word:
        # 部分嵌入值缺失
        if not len(real_emb) == 300:
            incomplete_num += 1
        elif real_word in w2v_dict.keys():
            duplicate_num += 1
        else:
            w2v_dict[real_word] = real_emb

print("incomplete word number is {}".format(incomplete_num))
print("duplicate word number is {}".format(duplicate_num))
json.dump(w2v_dict, open(output_path, "w"), ensure_ascii=False)