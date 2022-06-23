# coding=utf-8

import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    
    # bert_model_path = 'bert-base-chinese'  # RoBert_large 路径
    # # hfl / chinese - roberta - wwm - ext
    # file_path = 'datasets/split_data'
    # BATCH_SIZE = 16
    # ENT_CLS_NUM = 1
    # EPOCH = 10
    
    # 模型基本参数类
    parser.add_argument("--bert_model_path", default=None, type=str, required=True,
                        help="bert_model_path")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="file_path")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="batch_size")
    parser.add_argument("--ent_cls_num", default=1, type=int,
                        help="ENT_CLS_NUM")
    parser.add_argument("--epoch", default=1, type=int,
                        help="EPOCH")
    parser.add_argument("--max_grad_norm", default=0.25, type=float, help="Max gradient norm.")
    parser.add_argument("--encoder_learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--decoder_learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--inner_dim", default=64, type=int,
                        help="the dimension of the output of two ffn layer after bert")
    parser.add_argument("--use_lstm", action="store_true",
                        help="Whether to add a bi-lstm above Bert.")
    parser.add_argument("--output_bar", default=0, type=float,
                        help="The bar over which the span will be output")
    
    # 程序类
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--get_badcase", action="store_true",
                        help="Whether to save bad cases")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the dev set.")
    
    # 比赛新增 trick
    parser.add_argument("--features", default="", type=str,
                        help="in_dic: whether the word is in dictionary, if yes set 1, else 0"
                             "w2v_emb: whether to concat corresponding word w2v embedding"
                             "flag_id: concat the one hot jieba flag emb after word embedding")
    parser.add_argument("--do_enhance", action="store_true",
                        help="Whether to use enhanced training data for fine tuning.")
    parser.add_argument("--do_rdrop", action="store_true",
                        help="Whether to use r-dropout when training.")
    parser.add_argument("--rdrop_alpha", default=4., type=float,
                        help="The alpha that control the kl loss when sum")
    parser.add_argument("--do_fgm", action="store_true",
                        help="Whether to FGM for advised learning.")
    parser.add_argument("--fgm_epsilon", default=0.5, type=float,
                        help="The epsilon of fgm")
    parser.add_argument("--do_swa", action="store_true",
                        help="Whether to use swa to average parameters.")
    
    # boundary smoothing: Boundary Smoothing for Named Entity Recognition
    parser.add_argument("--do_boundary_smoothing", action="store_true",
                        help="Whether to do boundary_smoothing.")
    parser.add_argument("--allow_single_token", action="store_true",
                        help="When boundary_smoothing, whether to let single token as one soft entity")
    parser.add_argument("--sb_epsilon", default=0.2, type=float,
                        help="The smoothing parameter, in source code is {.1, .2, .3}")
    parser.add_argument("--sb_size", default=1, type=int,
                        help="The smoothing parameter,  in source code is {1, 2}")
    parser.add_argument("--sb_adj_factor", default=0.2, type=float,
                        help="Dot smoothing probability, in source code is 1.0 ")
    
    return parser
