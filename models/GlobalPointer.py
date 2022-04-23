# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()
    
    def get_sample_f1(self, y_pred, y_true):
        """
        y_true: (batch_size, ent_type_size, seq_len, seq_len)
        y_pred: (batch_size, ent_type_size, seq_len, seq_len)
        """
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)
    
    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)
    
    def get_evaluate_fpr(self, y_pred, y_true, bar=0.):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > bar)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        if Y == 0 or Z == 0:
            return 0, 0, 0
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


class RawGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE
    
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device
        
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]
        
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        
        return logits / self.inner_dim ** 0.5


class SinusoidalPositionEmbedding(nn.Module):
    """ 定义Sin-Cos位置Embedding
    """
    
    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim  # 输出维度
        self.merge_mode = merge_mode  # 融合方式
        self.custom_position_ids = custom_position_ids
    
    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            #  inputs: [batch_size, seq_len, inner_dim*2]
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)  # d
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)  # \theta
        embeddings = torch.einsum('bn,d->bnd', position_ids,
                                  indices)  # [batch_size, seq_len], [seq_len//2] ->[batch_size, seq_len, seq_len//2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class EffiGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, feature_size, RoPE=True, use_lstm=False):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super(EffiGlobalPointer, self).__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        
        self.feature_size = feature_size
        self.hidden_size = encoder.config.hidden_size + feature_size
        self.RoPE = RoPE
        self.output_dropout = nn.Dropout(0.1)
        
        self.lstm = None
        if use_lstm:
            # 是否加lstm
            #     是：768 or 768 + feature_dim ---> 768 (encoder.config.hidden_size)
            #     否：768 or 768 + feature_dim 不变 (self.hidden_size)
            from torch.nn import LSTM
            self.lstm = LSTM(
                self.hidden_size,  # 输入为 bert_emb + hand_feature_emb
                self.encoder.config.hidden_size // 2,  # 输出为 bert_emb
                batch_first=True,
                bidirectional=True
            )
            # [ 768 + feature_size, 64*2 ]
            self.dense_1 = nn.Linear(encoder.config.hidden_size, self.inner_dim * 2)
            # [ 768 + feature_size, 9*2 ]
            self.dense_2 = nn.Linear(encoder.config.hidden_size,
                                     self.ent_type_size * 2)
        else:
            # [ 768 + feature_size, 64*2 ]
            self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
            # [ 768 + feature_size, 9*2 ]
            self.dense_2 = nn.Linear(self.hidden_size,
                                     self.ent_type_size * 2)
    
    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            # mask拓展axis个维度
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)  # [b, 1, 1, s]
            # 如果还是不对齐，则在mask后面拓展维度
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)
    
    def add_mask_tril(self, logits, mask):
        """
        logits: [ b, t_n, s, s ]
        mask: [ b , s ]
        """
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits
    
    def forward(self, input_ids, attention_mask, token_type_ids, hand_features):
        self.device = input_ids.device
        # encoder 得到 h
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state  # [b, s, h]
        
        # 是否拼接特征 最终维度为 768 or 768 + feature_dim
        if self.feature_size == 0:
            add_features_emb = last_hidden_state
        else:
            # hand_features = hand_features.view((hand_features.shape[0], hand_features.shape[1], self.feature_size))
            add_features_emb = torch.cat((last_hidden_state, hand_features), dim=2)
        
        # add_features_emb = self.output_dropout(add_features_emb)
        
        if self.lstm is not None:
            # residual = sequence_output  # 加了特征后做不了残差了 ...(但是说不定可以让 bert 输出当作残差)
            # lstm_output, (_, _) = self.lstm(sequence_output)
            # sequence_output = residual + lstm_output
            # add_features_emb = self.output_dropout(lstm_output)
            
            lstm_output, (_, _) = self.lstm(add_features_emb)
            add_features_emb = self.output_dropout(lstm_output)
        
        outputs = self.dense_1(add_features_emb)  # [ b, s, inner_dim*2 ]
        qw, kw = outputs[..., ::2], outputs[..., 1::2]  # 从0,1开始间隔为2
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        
        # 第一项： [b, s, s]
        # 用 self.inner_dim ** 0.5 作为分母放缩了一下
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        
        # 第二项： [b, type_num*2, s]
        bias = torch.einsum('bnh->bhn', self.dense_2(add_features_emb)) / 2
        
        # 分别拓展维度： [b, 1, s, s] + [b, t_n, 1, s] + [b, t_n, s, 1]
        # 广播后：      [b, t_n, s, s]
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 增加一个维度
        # mask
        logits = self.add_mask_tril(logits, mask=attention_mask)
        return logits
