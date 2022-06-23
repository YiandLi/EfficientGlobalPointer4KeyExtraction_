# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: 实体抽取.
"""
import json
import os
import random
import sys
import warnings
import numpy

warnings.filterwarnings("ignore")
sys.path.append("./")
from src.predict_CME import predict, get_bad_cases
from utils.data_loader import EntDataset
from transformers import BertTokenizerFast, BertModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchcontrib.optim import SWA
from models.GlobalPointer import EffiGlobalPointer, MetricsCalculator
from tqdm import tqdm
from utils.finetuning_argparse import get_argparse
from utils.logger import logger
from utils.bert_optimization import BertAdam
from utils.ths_data_utills import read_example_form_file

args = get_argparse().parse_args()
for i, j in args.__dict__.items():
    logger.info("{}:\t{}".format(i, j))
logger.info("======================================")


def set_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


set_seed(421)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# tokenizer
# BertTokenizerFast:
tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_path, do_lower_case=True)

# train_data and val_data
if args.do_enhance:
    train_eamples = read_example_form_file(args, 'enhanced_train')
else:
    train_eamples = read_example_form_file(args, 'train')

ner_train = EntDataset(args, train_eamples, tokenizer=tokenizer)
ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, collate_fn=ner_train.collate, shuffle=True,
                              num_workers=16 if device == 'cuda' else 0
                              )
dev_eamples = read_example_form_file(args, 'dev')
ner_evl = EntDataset(args, dev_eamples, tokenizer=tokenizer)
ner_loader_evl = DataLoader(ner_evl, batch_size=args.batch_size, collate_fn=ner_evl.collate, shuffle=False,
                            num_workers=16 if device == 'cuda' else 0
                            )

# GP MODEL
"""
encoder 使用 Huggingface transformers 加载：https://github.com/Langboat/Mengzi
tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base")
model = BertModel.from_pretrained("Langboat/mengzi-bert-base")
"""
encoder = BertModel.from_pretrained(args.bert_model_path)
logger.info("total_feature_size = {}\n".format(args.total_feature_size))
model = EffiGlobalPointer(encoder, ent_type_size=args.ent_cls_num, inner_dim=args.inner_dim,
                          feature_size=args.total_feature_size, use_lstm=args.use_lstm).to(device)  # 9个实体类型


# optimizer
def set_optimizer(model, train_steps=None):
    # param_optimizer = list(model.named_parameters())
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=0.1,
    #                      t_total=train_steps)
    
    # encoder 的 named_parameters()
    encoder_param_optimizer = list(model.encoder.named_parameters())
    
    # classifier/dense 的 named_parameters()
    dense_para_optimizer = list(model.dense_1.named_parameters())
    dense_para_optimizer.extend(list(model.dense_2.named_parameters()))
    if args.use_lstm:
        dense_para_optimizer.extend(list(model.lstm.named_parameters()))
    
    # 筛选 pooler
    encoder_param_optimizer = [n for n in encoder_param_optimizer if 'pooler' not in n[0]]
    dense_para_optimizer = [n for n in dense_para_optimizer if 'pooler' not in n[0]]
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters += [
        {'params': [p for n, p in encoder_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01,
         'lr': args.encoder_learning_rate},
        {'params': [p for n, p in encoder_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.encoder_learning_rate}
    ]
    optimizer_grouped_parameters += [
        {'params': [p for n, p in dense_para_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01,
         'lr': args.decoder_learning_rate
         },
        {'params': [p for n, p in dense_para_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.decoder_learning_rate}
    ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.encoder_learning_rate,
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer


total_tarin_step = (int(len(ner_train) / args.batch_size) + 1) * args.epoch
optimizer = set_optimizer(model, train_steps=total_tarin_step)
if args.do_swa:
    start_step = total_tarin_step * 3 // 4
    frep_step = int(len(ner_train) / args.batch_size)
    optimizer = SWA(optimizer, swa_start=start_step, swa_freq=frep_step, swa_lr=4e-5)
    logger.info("Setting SWA with swa_start:{}, swa_freq:{}, swa_lr:{}".format(optimizer.swa_start,
                                                                               optimizer.swa_freq,
                                                                               optimizer.swa_lr))


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    y_pred: (batch_size * ent_type_size, seq_len * seq_len)
    y_true: (batch_size * ent_type_size, seq_len * seq_len)
    
    两种方式引入  boundary smoothing，exp前或者exp后，这里使用前者，因为后面再加exp，仍然满足概率计算的方式
    """
    y_true_mask = (y_true > 0).long()
    y_pred = (1 - 2 * y_true_mask) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true_mask * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true_mask) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])  # (batch_size * ent_type_size, 1)
    
    # 增加 boundary smoothing 部分
    y_pred_pos = y_pred_pos * y_true
    y_pred_pos[y_pred_pos == 0] = -1e12
    # 都加1列 0 ，exp(0)=1 表示 1
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def og_multilabel_categorical_crossentropy(y_pred, y_true):
    """
    y_pred: (batch_size * ent_type_size, seq_len * seq_len)
    y_true: (batch_size * ent_type_size, seq_len * seq_len)
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])  # (batch_size * ent_type_size, 1)
    # 都加1列 0 ，exp(0)=1 表示 1
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def loss_fun(y_pred, y_true):
    """
    y_true: (batch_size, ent_type_size, seq_len, seq_len)
    y_pred: (batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_pred, y_true)
    return loss


def r_drop_loss(logits_1, logits_2, y_true, alpha):
    loss_circle = loss_fun(logits_1, y_true) + loss_fun(logits_2, y_true)
    # print(torch.sum(logits_1-logits_2))
    # exit()
    batch_size, ent_type_size = logits_1.shape[:2]
    logits_2 = logits_2.reshape(batch_size * ent_type_size, -1)
    logits_1 = logits_1.reshape(batch_size * ent_type_size, -1)
    loss_kl = F.kl_div(F.log_softmax(logits_1, dim=-1), F.softmax(logits_2, dim=-1), reduction='sum') + \
              F.kl_div(F.log_softmax(logits_2, dim=-1), F.softmax(logits_1, dim=-1), reduction='sum')
    loss_circle /= 2
    loss_kl /= 2
    return loss_circle + loss_kl * alpha


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    
    def attack(self, epsilon=args.fgm_epsilon, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if args.do_train:
    # save args
    args_path = "checkpoints/args.json"
    json.dump(args.__dict__, open(args_path, "w"))
    
    metrics = MetricsCalculator()
    
    max_f, max_recall, max_epoch = 0.0, 0.0, 0
    for eo in range(args.epoch):
        total_loss, total_f1 = 0., 0.
        for idx, batch in enumerate(ner_loader_train):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, hand_features = batch
            input_ids, attention_mask, segment_ids, labels, hand_features = \
                input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(
                    device), hand_features.to(device)
            
            # r-dropout 和 ce 对应
            if args.do_rdrop:
                logits = model(input_ids, attention_mask, segment_ids, hand_features)
                set_seed(122)  # 这个地方怎么不管用了 我去！！！
                # [ batch, class_num, seq_len, seq_len ]
                logits_rd = model(input_ids, attention_mask, segment_ids, hand_features)
                set_seed(42)
                loss = r_drop_loss(logits, logits_rd, labels, args.rdrop_alpha)
            else:
                logits = model(input_ids, attention_mask, segment_ids, hand_features)
                loss = loss_fun(logits, labels)
            optimizer.zero_grad()  # 清空上一轮梯度，再进行 backward() 赋予梯度
            loss.backward()
            
            # 对抗损失
            if args.do_fgm:
                fgm = FGM(model)  # 初始化
                fgm.attack()  # 在embedding上添加对抗扰动
                logits_fgm = model(input_ids, attention_mask, segment_ids, hand_features)
                loss_adv = loss_fun(logits_fgm, labels)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数
                # optimizer.step()  # 梯度下降，更新参数，在后面也会做，这里舍去
                # model.zero_grad()  # 等价于上面的 optimizer.zero_grad()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            sample_f1 = metrics.get_sample_f1(logits, labels)
            total_loss += loss.item()
            total_f1 += sample_f1.item()
            
            avg_loss = total_loss / (idx + 1)
            avg_f1 = total_f1 / (idx + 1)
            
            if args.do_swa:
                optimizer.swap_swa_sgd()  # 重置权重
            
            if idx % 10 == 0:
                logger.info("trian_loss:%f\t train_f1:%f" % (avg_loss, avg_f1))
        
        with torch.no_grad():
            total_f1_, total_precision_, total_recall_ = 0., 0., 0.
            model.eval()
            for batch in tqdm(ner_loader_evl, desc="Valing"):
                raw_text_list, input_ids, attention_mask, segment_ids, labels, hand_features = batch
                input_ids, attention_mask, segment_ids, labels, hand_features = input_ids.to(device), attention_mask.to(
                    device), segment_ids.to(device), labels.to(device), hand_features.to(device)
                logits = model(input_ids, attention_mask, segment_ids, hand_features)
                f1, p, r = metrics.get_evaluate_fpr(logits, labels, bar=args.output_bar)
                total_f1_ += f1
                total_precision_ += p
                total_recall_ += r
            avg_f1 = total_f1_ / (len(ner_loader_evl))
            avg_precision = total_precision_ / (len(ner_loader_evl))
            avg_recall = total_recall_ / (len(ner_loader_evl))
            logger.info("EPOCH:%d\t EVAL_F1:%f\tPrecision:%f\tRecall:%f\t" % (eo, avg_f1, avg_precision, avg_recall))
            if avg_f1 > max_f:
                torch.save(model.state_dict(), 'checkpoints/ths_model.pth')
                # 保存args
                # args_path = "checkpoints/args.json"
                # json.dump(args.__dict__, open(args_path, "w"))
                max_f = avg_f1
                max_epoch = eo
            model.train()
    logger.info("best f1 scaore is {} in epoch {}".format(max_f, max_epoch))

if args.get_badcase:
    get_bad_cases(args)

if args.do_predict:
    predict(args)
