# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 13:52
@Auth: Xhw
@Description: CMeEE实体识别的数据载入器
"""
import json
from typing import Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

# max_len = 256  # CMeEE
# ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}  # CMeEE

# 同花顺
max_len = 32
ent2id = {"PHRASE": 0}  # CMeEE
id2ent = {}
for k, v in ent2id.items(): id2ent[v] = k


def load_data(path):
    """
    返回数据结构：
        [
            [text, (start_idx, end_idx, label_id), (start_idx, end_idx, label_id), ...] ,
        ... ]
    """
    D = []
    for d in json.load(open(path)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, ent2id[label]))  # CMeEE
    return D


class EntDataset(Dataset):
    def __init__(self, args, data, tokenizer, istrain=True):
        self.data = data
        self.tokenizer = tokenizer
        self.istrain = istrain
        self.args = args
    
    def __len__(self):
        return len(self.data)
    
    def encoder(self, item):
        """
        item 是每一个instance字典
        """
        if self.istrain:
            text = item["sentence"]
            """
            BertTokenizerFast inherits from PreTrainedTokenizerFast
                return_offsets_mapping: Whether or not to return (char_start, char_end) for each token. 返回索引位置（空格也算
                
                tokenizer('I have the biggest ideas to mapping.', return_offsets_mapping=True)
                Out[5]: {'input_ids': [101, 1045, 2031, 1996, 5221, 4784, 2000, 12375, 1012, 102],
                        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        'offset_mapping': [(0, 0), (0, 1), (2, 6), (7, 10), (11, 18), (19, 24), (25, 27), (28, 35), (35, 36), (0, 0)]}
                        
                tokenizer.tokenize('I have the biggest ideas to mapping.')
                Out[6]: ['i', 'have', 'the', 'biggest', 'ideas', 'to', 'mapping', '.']
                
                start_mapping = { start_idx : token_id, .. }
                              = { 0:1, 2:2, 7:3, 11:4, 19:5, 25:6, 28:7, 35:8 }
                              
                end_mapping   = { end_idx   : token_id, .. }
                              = { 0:1, 5:2, 9:3, 17:4, 23:5, 26:6, 34:7, 35:8 }
                              
                              
                important apple          im       ##port  ##ant    apple
                (10,20)   (21,22)        (10,11) (12,17)  (18,20)  (21,22)
                                         (10,1)   .....            (21,4)
                                         (11,1)   .....            (22,4)
                (10,22,n)                (1, 4, n)
            """
            # 切除最终sub-token的 （start_idx, end_idx）
            token2char_span_mapping = \
                self.tokenizer(text, return_offsets_mapping=True, max_length=max_len, truncation=True)["offset_mapping"]
            # { 每个token的开始字符的索引: 第几个token } and { end_index : token_index }
            start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            # 将raw_text的下标 与 token的start和end下标对应
            encoder_txt = self.tokenizer.encode_plus(text, max_length=max_len, truncation=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]
            
            # feature 部分
            feature_embs = torch.empty([len(input_ids), 0])  # 后面加feature的话还是这个，但是是纵向加
            if "in_dic" in self.args.features:
                """
                   text：Gep 是什么
                   entity：Gep
                   分词结果：                         【unk】    是     什        么
                   dic_feature：                     [1,1,1,   0,     0,       0]    注意Gep没有切分，是按位置对应1的
                   token2char_span_mapping：[(0, 0), (0, 3), (4, 5), (5, 6), (6, 7), (0, 0)] 而这里被切开了
    
                   注意前后有 (0,0) := [cls],[sep]，token2char_span_mapping长度和token_type_ids/attention_mask长度相同
                   所以特征向量开头和结束要填充 0
                   其余部位，如果token2char_span_mapping中元两组个元素位置都对应dic_feature（在字典中）为1，则填充subtoken为1；有一个1则填充0；没有则填充0
               """
                dic_feature = item["dic_features"]
                dic_embs = []
                for s_idx, e_idx in token2char_span_mapping:  # token2char_span_mapping 就对应着input_ids/subtoken，所以长度一致
                    if (s_idx, e_idx) == (0, 0):
                        dic_embs.append(0)
                    else:
                        if dic_feature[s_idx] == 1 and dic_feature[e_idx - 1] == 1:
                            dic_embs.append(1)
                        elif dic_feature[s_idx] == 1 or dic_feature[e_idx - 1] == 1:
                            dic_embs.append(0.5)
                        else:
                            dic_embs.append(0)
                assert len(dic_embs) == len(input_ids)
                feature_embs = torch.hstack([feature_embs, torch.tensor(dic_embs)[:, None]])
            if "w2v_emb" in self.args.features:
                in_w2v_feature = item["w2v_features"]
                w2v_emb = len(in_w2v_feature[0])
                out_w2v_feature = []
                for s_idx, e_idx in token2char_span_mapping:
                    if (s_idx, e_idx) == (0, 0):
                        # cls 和 sep 都填充 0
                        out_w2v_feature.append(torch.tensor([float(0) for i in range(w2v_emb)]))
                    elif e_idx - s_idx == 1:  # 逐字切分
                        out_w2v_feature.append(torch.tensor(in_w2v_feature[s_idx]))
                    else:  # 这里如果切成 "SQ13"（一个token），使用 max pooling，得到token
                        word_pooling = [in_w2v_feature[single_one] for single_one in range(s_idx, e_idx)]
                        word_pooling = torch.tensor(word_pooling)
                        word_pooling = torch.amax(word_pooling, 0)
                        out_w2v_feature.append(word_pooling)
                assert len(out_w2v_feature) == len(input_ids)
                feature_embs = torch.hstack(
                    [feature_embs, torch.tensor([item.detach().numpy() for item in out_w2v_feature])])
            if "flag_id" in self.args.features:
                in_flag_feature = item["flag_features"]
                flag_num = len(in_flag_feature[0])
                out_flag_feature = []
                for s_idx, e_idx in token2char_span_mapping:
                    if (s_idx, e_idx) == (0, 0):
                        # cls 和 sep 都填充 0
                        out_flag_feature.append(torch.tensor([float(0) for i in range(flag_num)]))
                    elif e_idx - s_idx == 1:
                        out_flag_feature.append(torch.tensor(in_flag_feature[s_idx]))
                    else:
                        flag_pooling = [in_flag_feature[single_one] for single_one in range(s_idx, e_idx)]
                        flag_pooling = torch.tensor(flag_pooling, dtype=float)  # 转化 tensor
                        flag_pooling = torch.mean(flag_pooling, 0)
                        #         或者 torch.sum(flag_pooling, 0) / len(flag_pooling)
                        out_flag_feature.append(flag_pooling)
                feature_embs = torch.hstack(
                    [feature_embs, torch.tensor([item.detach().numpy() for item in out_flag_feature])])
                # 再确保下为 float32 ，后面predict时会变成 float64，这里没有出现，但是还是加上去吧。
                feature_embs.float()
            return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, feature_embs
        else:
            # TODO 测试
            pass
    
    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        seq_dims : 控制维度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs],
                            axis=0)  # label_num, max_seq_len, max_seq_len，注意这里 max_seq_len 是同batch内最长句子的长度
        elif not hasattr(length, '__getitem__'):
            length = [length]
        
        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        
        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            
            # pad_width是在各维度的各个方向上想要填补的长度,如（（1，2），（2，2））
            # 表示在第一个维度上水平方向上padding=1,垂直方向上padding=2
            # 在第二个维度上水平方向上padding=2,垂直方向上padding=2。
            # 如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)
        
        return np.array(outputs)
    
    def get_boundary_smoothing(self, start, end, label, labels, real_length, allow_single_token=True):
        """
        labels:  entity_num, max_len, max_len 注意本代码中本参数是截断过的
        real_length: 真实的样本长度
        allow_single:  默认单个token不算作entity
        
        采用累加的方式 ，逐个 golden entity 遍历
        """
        
        def _spans_from_surrounding(span: Tuple[int], distance: int, num_tokens: int, allow_single_token: bool):
            """Spans from the surrounding area of the given `span`.
            """
            for k in range(distance):
                for start_offset, end_offset in [(-k, -distance + k),
                                                 (-distance + k, k),
                                                 (k, distance - k),
                                                 (distance - k, -k)]:
                    start, end = span[0] + start_offset, span[1] + end_offset
                    # 源码为  0<=start < end <= num_tokens 但是个人 觉得 cls 和 sep 不应该被计算在内，一定不是实体
                    # num_token = seq_len  + 2 (cls+sep)
                    # 0    1      2      3
                    # [cls token1 token2 sep]    长度为4 但是实际上索引应该小于3(sep)，所以是 end < num_tokens-1
                    if allow_single_token and 0 < start <= end < num_tokens - 1:
                        yield (start, end)
                    elif 0 < start < end < num_tokens - 1:
                        yield (start, end)
        
        labels[label, start, end] += (1 - self.args.sb_epsilon)
        for dist in range(1, self.args.sb_size + 1):  # 遍历距离
            eps_per_span = self.args.sb_epsilon / (self.args.sb_size * dist * 4)
            sur_spans = list(
                _spans_from_surrounding((start, end), dist, real_length, self.args.allow_single_token))  # 找到所有曼哈顿距离为dist的「周围span」的坐标
            for sur_start, sur_end in sur_spans:
                labels[label, sur_start, sur_end] += (
                        eps_per_span * self.args.sb_adj_factor
                )
            labels[label, start, end] += eps_per_span * (dist * 4 - len(sur_spans))
        
        return labels
    
    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        batch_hand_features = []
        
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, feature_embs \
                = self.encoder(item)
            # 得到目标概率矩阵
            labels = np.zeros((len(ent2id), max_len, max_len))  #
            # labels_binary = np.zeros((2, max_len, max_len))  # 只考虑是否为 entity
            for start, end, label in item["entities"]:  # 全都是index索引，label不用额外转换
                if start in start_mapping and end in end_mapping:
                    # 得到概率矩阵中的 横竖索引
                    # 其实就是span 的 start_idx 和 end_idx 控制的位置
                    start = start_mapping[start]
                    end = end_mapping[end]
                    if not self.args.do_boundary_smoothing:
                        labels[label, start, end] = 1  # [label_num, seq_len, seq_len]
                    else:
                        labels = self.get_boundary_smoothing(start, end, label, labels, len(input_ids))
                
                # for end_ in range(start, end):
                #     labels_binary[1, start, end_] = 1
                # labels_binary[1, start, end] = 1
            
            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
            batch_hand_features.append(feature_embs)
        
        # padding
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        # batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3))
        # 这里出现 bug，即long()，让所有数值（0.8, 0.1）全部变换为0，同时boundary并不适用于circle loss
        
        
        # feature
        batch_handfeatures = torch.tensor(self.sequence_padding(batch_hand_features, seq_dims=2)).float()
        return raw_text_list, batch_inputids, batch_attentionmask, batch_segmentids, batch_labels, batch_handfeatures  # , batch_labels_bin
    
    def __getitem__(self, index):
        item = self.data[index]
        return item
