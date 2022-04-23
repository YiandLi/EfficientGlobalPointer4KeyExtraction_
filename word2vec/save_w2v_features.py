"""
保存每个实例的对应单词，flag，和对应单词的词向量
1. 得到字典和语料
2. 对每个实例的文本进行 jieba.cut 默认分词 （有条件的自助加词表）
3. 在字典中查找分词的 embedding
4. 存储 分词，词性和 embedding

sample:
    input: "我特别喜欢吃苹果。"
    word_features： [ 我, 特别, 特别, 喜欢, 喜欢, 吃， 苹果， 苹果 ]
    flag_features： [ r,  d,   d,    v,   v,   v,  n,    n   ]   一共23
    emb_features：  [ [我_emb], [特别_emb], [特别_emb], ... ]
    
分词方式：
    1. jieba.cut 默认分词
    2. THUOCL (格式问题，使用 thulac 加载)
"""

import jieba.posseg as pseg
import json


def transfer_and_save(mode, flag_dic):
    word_to_save = []
    emb_to_save = []
    flag_to_save = []
    
    # 被转换的目标语料
    data_path = "../datasets/split_data/{}.json".format(mode)
    data = json.load(open(data_path, "r"))
    
    # 遍历切词，查询
    for i, j in enumerate(data):
        text = j[0]
        text_cut = pseg.cut(text)
        word_features = []
        emb_features = []
        flag_features = []
        for word in text_cut:
            for _ in range(len(word.word)):
                # 存 word
                word_features.append(word.word)
                # 存 w2v embedding
                try:
                    emb_features.append([float(i) for i in w2v_dict[word.word]])
                except:
                    emb_features.append([])
                # 存 flag
                flag_features.append(word.flag)
        # 存所有 flag
        flag_dic.extend(flag_features)
        flag_dic = list(set(flag_dic))
        
        assert len(word_features) == len(text)
        word_to_save.append(word_features)
        emb_to_save.append(emb_features)
        flag_to_save.append(flag_features)
    
    # 存储
    json.dump(word_to_save,
              open("../datasets/split_data/features/word_feature/{}_word_features.json".format(mode), "w"),
              ensure_ascii=False)
    json.dump(flag_to_save,
              open("../datasets/split_data/features/word_feature/{}_flag_features.json".format(mode), "w"),
              ensure_ascii=False)
    json.dump(emb_to_save,
              open("../datasets/split_data/features/word_feature/{}_word_emb_features.json".format(mode), "w"),
              ensure_ascii=False)
    
    return flag_dic


if __name__ == "__main__":
    # w2v 字典
    dict_path = "data/word2vec_dic.json"
    w2v_dict = json.load(open(dict_path, "r"))
    flag_dic = []
    
    # 遍历数据集
    for i in ["train", "dev", "test"]:
        flag_dic = transfer_and_save(i, flag_dic)
    
    # 保存flag字典
    flag2id = {j: i for i, j in enumerate(flag_dic)}
    json.dump(flag2id,
              open("../datasets/split_data/features/word_feature/flag2id.json", "w"),
              ensure_ascii=False)
