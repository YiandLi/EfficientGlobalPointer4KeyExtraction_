import json
import os
import torch
from utils.logger import logger


class InputExample(object):
    """A single training/test example for token classification."""
    
    def __init__(self, guid, words, labels):
        """Construct a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            apecified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
    
    def __repr__(self):
        # print(类的实例)：__repr__ 规定的内容
        return json.dumps(self.__dict__, ensure_ascii=False)


def read_example_form_file(args, mode):
    """
    读取「同花顺比赛」数据集，打包并返回为 InputExample 列表

    :param data_dir: 数据集地址
    :param mode: train / dev / test
    :return:
        [
            {"sentence" : sentence,
             "entities" : [(start_idx, end_idx, label_id), (start_idx, end_idx, label_id), ...] ,
             "dic_features" : [0,0,0,1,...]
            }
        ... ]
    """
    
    # mrc格式
    file_path = os.path.join(args.data_dir, mode + ".json")
    all_data = json.load(open(file_path, encoding="utf-8"))
    
    # 特征
    args.total_feature_size = 0
    if "in_dic" in args.features:
        dic_path = os.path.join(args.data_dir, "features/dic/all_dic.json")
        dic = json.load(open(dic_path, "r"))
        args.total_feature_size += 1
    if "w2v_emb" in args.features:
        word_emb_path = os.path.join(args.data_dir, "features/word_feature/{}_word_emb_features.json".format(mode))
        word_emb_dict = json.load(open(word_emb_path, "r"), )  # strict=False
        args.total_feature_size += 300
    if "flag_id" in args.features:
        flag_feature_path = os.path.join(args.data_dir, "features/word_feature/{}_flag_features.json".format(mode))
        flag_features = json.load(open(flag_feature_path, "r"))
        flag_id_path = os.path.join(args.data_dir, "features/word_feature/flag2id.json")
        flag_dic = json.load(open(flag_id_path, "r"))
        flag_num = len(flag_dic)
        args.total_feature_size += flag_num
    
    # 存储
    examples = []
    for i, (sentence, entity_list) in enumerate(all_data):
        examples.append({"sentence": sentence, "entities": []})
        # 得到labels列表
        for entity in entity_list:
            start_idx = sentence.find(entity)
            end_idx = start_idx + len(entity) - 1
            examples[-1]["entities"].append((start_idx, end_idx, 0))
        # 需要保证 word_emb_dict 对应 sentence
        if "in_dic" in args.features:
            in_dict_feature = torch.zeros(len(sentence))
            for word in dic:
                if word in sentence:
                    index_list = str_all_index(sentence, word)
                    for idx in index_list:
                        in_dict_feature[idx: idx + len(word)] = 1.
            examples[-1]["dic_features"] = in_dict_feature
        if "w2v_emb" in args.features:
            # 有些字典没有的，填充 0
            for idx, word_emb in enumerate(word_emb_dict[i]):
                if not word_emb:
                    word_emb_dict[i][idx] = [float(0) for i in range(300)]
            examples[-1]["w2v_features"] = word_emb_dict[i]
        if "flag_id" in args.features:
            flag_list = flag_features[i]
            for idx, flag in enumerate(flag_list):
                flag_list[idx] = get_one_hot_list(flag_num, flag_dic[flag])
                assert 1 in flag_list[idx]
            examples[-1]["flag_features"] = flag_list
    
    return examples


def get_one_hot_list(total_class_num: int, target_idx: int):
    empty = [0] * total_class_num
    empty[target_idx] = 1
    return empty


def save_predictions(all_data, file_to_predict):
    """
    all_dicts:
        [   {   'text': '机构评级最高的股票是',
                'entities': [ {'start_idx': 0, 'end_idx': 0, 'type': 'PHRASE'},
                              {'start_idx': 0, 'end_idx': 1, 'type': 'PHRASE'}
                             ]
             }, { ... }, ...
        ]
    """
    with open(file_to_predict, "w") as f:
        f.write("用户问句\t名词短语")
        for i, d in enumerate(all_data):
            # i, (sentence, entity_list)
            sentence = d['text']
            phases = []
            for entity in d['entities']:
                phases.append(sentence[entity["start_idx"]:entity["end_idx"] + 1])
            phases = "_|_".join(phases)
            f.write("\n%s\t%s" % (sentence, phases))
    
    logger.info("Writing predictions in %s", file_to_predict)


def str_all_index(str_, a):
    """
    在字符串 str_ 中找到所有字串 a 的起始索引
    Return: index_list : list
    首先输入变量2个，输出list，然后中间构造每次find的起始位置start,start每次都在找到的索引+1，后面还得有终止循环的条件
    """
    index_list = []
    start = 0
    while True:
        x = str_.find(a, start)
        if x > -1:
            start = x + 1
            index_list.append(x)
        else:
            break
    return index_list
