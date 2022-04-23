import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("./")

import numpy
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from models.GlobalPointer import EffiGlobalPointer, MetricsCalculator
from transformers import BertTokenizerFast, BertModel
from utils.logger import logger
from utils.ths_data_utills import read_example_form_file, save_predictions


def set_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_argparse():
    parser = argparse.ArgumentParser()
    # 模型基本参数类
    parser.add_argument("--checkpoints", default=None, required=True, nargs='+',
                        help="list of bert_model_path")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="file_path")
    return parser


def read_args(args_path):
    if not os.path.exists(args_path):
        raise ValueError(args_path, " is not found")
    with open(args_path) as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
    return args


def get_spans(item, tokenizer):
    token2char_span_mapping = tokenizer(item["sentence"], return_offsets_mapping=True, max_length=max_len)[
        "offset_mapping"]
    new_span = []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])
    return new_span


def get_logits(item, tokenizer, ner_model, max_len=None, device=None, id2ent=None, args=None):
    token2char_span_mapping = tokenizer(item["sentence"], return_offsets_mapping=True, max_length=max_len)[
        "offset_mapping"]
    
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
                # 或者 torch.sum(flag_pooling, 0) / len(flag_pooling)
                out_flag_feature.append(flag_pooling)
        feature_embs = torch.hstack(
            [feature_embs, torch.tensor([item.detach().numpy() for item in out_flag_feature])])
    
    feature_embs = torch.tensor(feature_embs).unsqueeze(0).float().to(device)
    
    scores = ner_model(input_ids, attention_mask, token_type_ids, feature_embs)[0].data.cpu().numpy()
    
    return scores


if __name__ == "__main__":
    set_seed(42)
    all_args = get_argparse().parse_args()
    for i, j in all_args.__dict__.items():
        logger.info("{}:\t{}".format(i, j))
    logger.info("======================================")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = 32
    ent_cls_num = 1
    ent2id = {"PHRASE": 0}
    id2ent = {}
    
    for k, v in ent2id.items():
        id2ent[v] = k
    
    all_outputs = []
    for i, model_path in enumerate(all_args.checkpoints):
        model_path = model_path.strip()
        # 读取args
        args_path = os.path.join(model_path, "args.json")
        args = read_args(args_path)
        
        # tokenizer = BertTokenizerFast.from_pretrained("../pre_ckpts/my_mengzi_6")
        # encoder = BertModel.from_pretrained("../pre_ckpts/my_mengzi_6")
        tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_path)
        encoder = BertModel.from_pretrained(args.bert_model_path)
        
        model = EffiGlobalPointer(encoder, ent_cls_num, args.inner_dim, feature_size=args.total_feature_size,
                                  use_lstm=args.use_lstm).to(device)
        model.load_state_dict(torch.load(model_path + "/ths_model.pth"
                                         , map_location=device
                                         ))
        logger.info("load the {} model from {}".format(str(i) + "st", model_path))
        model.eval()
        this_logistic = []
        test_examples = read_example_form_file(args, 'test')
        
        for d in test_examples:
            # 添加一个样本的 logistic
            this_logistic.append(  # ndarray 的 list [batch_size, class_num(1), seq_len, seq_len] （ 不同样本的seq_len长度不同
                get_logits(d, tokenizer=tokenizer, max_len=max_len,  # [1, seq_len, seq_len ]
                           ner_model=model, device=device, id2ent=id2ent, args=args))
        all_outputs.append(this_logistic)  # model_num, batch_size, class_num(1), seq_len, seq_len
    
    # 得到所有样本 span 集合
    all_spans = []
    for d in test_examples:
        all_spans.append(get_spans(d, tokenizer))
    
    # # 得到 eval 集合
    # dev_eamples = read_example_form_file(args, 'dev')
    # ner_evl = EntDataset(args, dev_eamples, tokenizer=tokenizer)
    # ner_loader_evl = DataLoader(ner_evl, batch_size=args.batch_size, collate_fn=ner_evl.collate, shuffle=False,
    #                             num_workers=16 if device == 'cuda' else 0
    #                             )
    
    # 求 test 集合平均
    logger.info("get avg of all instances")
    all_ = []
    for example_idx in tqdm(range(len(test_examples))):  # 遍历样本
        # todo：对每个样本，每个模型进行预测。
        instance_logistic = []
        for model_idx in range(len(all_args.checkpoints)):  # 遍历模型
            instance_logistic.append(all_outputs[model_idx][example_idx])  # 将单个不同模型单个样本的输出存入
        # todo：得到每个样本的 avg
        instance_logistic = numpy.array(instance_logistic)
        instance_logistic = numpy.average(instance_logistic, axis=0)
        # todo：处理样本，得到实体信息
        entities = []
        instance_logistic[:, [0, -1]] -= numpy.inf  # [cls] 不构成实体
        instance_logistic[:, :, [0, -1]] -= numpy.inf  # [sep] 不构成实体
        new_span = all_spans[example_idx]
        for l, start, end in zip(*numpy.where(instance_logistic > 0.)):
            entities.append({"start_idx": new_span[start][0], "end_idx": new_span[end][-1], "type": id2ent[l]})
        all_.append({"text": test_examples[example_idx]["sentence"], "entities": entities})
    save_predictions(all_, all_args.output_dir)
