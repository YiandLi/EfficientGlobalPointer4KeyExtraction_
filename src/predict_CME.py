# -*- coding: utf-8 -*-
"""
@Auth: Xhw
"""
from transformers import BertModel, BertTokenizerFast
from models.GlobalPointer import EffiGlobalPointer as GlobalPointer
import torch
import numpy as np
from tqdm import tqdm
from utils.logger import logger
from utils.ths_data_utills import read_example_form_file, save_predictions


def NER_RELATION(item, tokenizer, ner_model, max_len=None, device=None, id2ent=None, args=None):
    token2char_span_mapping = tokenizer(item["sentence"], return_offsets_mapping=True, max_length=max_len)[
        "offset_mapping"]
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])
    
    encoder_txt = tokenizer.encode_plus(item["sentence"], max_length=max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
    
    # 增加人工特征部分
    feature_embs = torch.empty([len(input_ids[0]), 0])
    if "in_dic" in args.features:
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
        feature_embs = torch.hstack([feature_embs, torch.tensor(dic_embs)[:, None]])
    if "w2v_emb" in args.features:
        in_w2v_feature = item["w2v_features"]
        out_w2v_feature = []
        for s_idx, e_idx in token2char_span_mapping:
            if (s_idx, e_idx) == (0, 0):
                # cls 和 sep 都填充 0
                out_w2v_feature.append(torch.tensor([float(0) for i in range(300)]))
            elif e_idx - s_idx == 1:
                out_w2v_feature.append(torch.tensor(in_w2v_feature[s_idx]))
            else:  # 这里如果切成 "QS13"（一个token），使用 max pooling，得到token
                word_pooling = [in_w2v_feature[single_one] for single_one in range(s_idx, e_idx)]
                word_pooling = torch.tensor(word_pooling)
                word_pooling = torch.amax(word_pooling, 0)
                out_w2v_feature.append(word_pooling)
        feature_embs = torch.hstack(
            [feature_embs, torch.tensor([item.detach().numpy() for item in out_w2v_feature])])
    if "flag_id" in args.features:
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
    
    feature_embs = torch.tensor(feature_embs).unsqueeze(0).float().to(device)
    
    scores = ner_model(input_ids, attention_mask,
                       token_type_ids, feature_embs)[0].data.cpu().numpy()  # [1, seq_len, seq_len ]
    
    scores[:, [0, -1]] -= np.inf  # [cls] 不构成实体
    scores[:, :, [0, -1]] -= np.inf  # [sep] 不构成实体
    for l, start, end in zip(*np.where(scores > args.output_bar)):
        entities.append({"start_idx": new_span[start][0], "end_idx": new_span[end][-1], "type": id2ent[l]})
    
    return {"text": item["sentence"], "entities": entities}


def predict(args):
    bert_model_path = args.bert_model_path  # your RoBert_large path
    save_model_path = 'checkpoints/ths_model.pth'  # 67.94%
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = 32
    ent_cls_num = 1
    ent2id = {"PHRASE": 0}
    id2ent = {}
    for k, v in ent2id.items():
        id2ent[v] = k
    
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
    encoder = BertModel.from_pretrained(bert_model_path)
    model = GlobalPointer(encoder, ent_cls_num, args.inner_dim, feature_size=args.total_feature_size,
                          use_lstm=args.use_lstm).to(device)
    model.load_state_dict(torch.load(save_model_path))
    logger.info("load model from {}".format(save_model_path))
    model.eval()
    
    all_ = []
    output_file = "result.txt"
    test_examples = read_example_form_file(args, 'test')
    for d in tqdm(test_examples):
        all_.append(
            NER_RELATION(d, tokenizer=tokenizer, max_len=max_len, ner_model=model, device=device, id2ent=id2ent,
                         args=args))
    """
    [   {   'text': '机构评级最高的股票是',
            'entities': [ {'start_idx': 0, 'end_idx': 0, 'type': 'PHRASE'},
                          {'start_idx': 0, 'end_idx': 1, 'type': 'PHRASE'}
                         ]
         }, { ... }, ...
    ]
    """
    save_predictions(all_, output_file)


def get_bad_cases(args):
    bert_model_path = args.bert_model_path  # your RoBert_large path
    save_model_path = 'checkpoints/ths_model.pth'  # 67.94%
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = 32
    ent_cls_num = 1
    ent2id = {"PHRASE": 0}
    id2ent = {}
    for k, v in ent2id.items(): id2ent[v] = k
    
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
    encoder = BertModel.from_pretrained(bert_model_path)
    model = GlobalPointer(encoder, ent_cls_num, args.inner_dim, feature_size=args.total_feature_size,
                          use_lstm=args.use_lstm).to(device)
    model.load_state_dict(torch.load(save_model_path))
    logger.info("load model from {}".format(save_model_path))
    model.eval()
    
    all_texts = []
    bc_results = []
    output_file = "../bad_cases.txt"
    for d in tqdm(read_example_form_file(args.file_path, 'dev')):
        text = d[0]
        all_texts.append(text)
        real_phases = []
        pred_phases = []
        for j in d[1:]:
            start_idx = j[0]
            end_idx = j[1]
            real_phases.append(text[start_idx: end_idx + 1])
        entities = NER_RELATION(text, tokenizer=tokenizer, max_len=max_len,
                                ner_model=model, device=device, id2ent=id2ent)['entities']
        """
        [ {'start_idx': 0, 'end_idx': 0, 'type': 'PHRASE'},
          ... ]
        """
        for entity in entities:
            pred_phases.append(text[entity["start_idx"]:entity["end_idx"] + 1])
        bc_results.append(compare_cases(real_phases, pred_phases))
    # 输出
    with open(output_file, "w") as f:
        f.write("用户问句\t未识别出短语\t识别错误短语")
        for sen, (un_rec, un_pre) in zip(all_texts, bc_results):
            if not un_pre and not un_rec:
                continue
            f.write("\n%s %s %s" % (sen, "｜".join(un_rec) if un_rec else "None",
                                    "｜".join(un_pre) if un_pre else "None"))


def compare_cases(label_items, pred_items):
    """
    :param label_items: [(2, 6, 'PHRASE'), (10, 13, 'PHRASE')] 正确实体元组
    :param pred_items: [(0, 4, 'PHRASE'), (0, 3, 'PHRASE'), (0, 2, 'PHRASE'), (0, 0, 'PHRASE'),..] 预测出实体元组
    :return:
    """
    un_rec = []
    un_pre = []
    for item in label_items:
        if item not in pred_items:
            un_rec.append(item)
    for item in pred_items:
        if item not in label_items:
            un_pre.append(item)
    return (un_rec, un_pre)
