# Coding
Original Model path：https://github.com/xhw205/Efficient-GlobalPointer-torch

Based on GlobalPointer，[Keras version](https://spaces.ac.cn/archives/8877) ，The solution key is to represent token-pair 。\
The base model is from a  torch version [repository](https://github.com/xhw205/GlobalPointer_torch)。

# Updating records
- 2022/04/23 buiild
- 2022/6/23 Add boundary smoothing function

# package dependencies
```
python==3.6
torch==1.8.1
transformers==4.4.1
```

I use a service for training ，the seting is：
```
python==3.7
torch==1.8.1
transformers==4.10.0
```
For more details, please refer to `server_requirements.txt`.

# File Structure
```
EfficientGlobalPointer4KeyExtraction
├── ensemble.sh  # model ensemble script
├── GP_runner.sh  # finetune  script
├── README.md
├── server_requirements.txt  # server dependencies
├── result.txt  
├── err.log
├── checkpoints  # model save dic
│   ├── bad_cases_GP.txt # badcase output
│   └── experiments_notes.md  # experiment record
├── datasets
│   └── split_data
│       ├── biaffine_labels.txt  # Entity tag file (entity type, without BIO)
│       ├── dev.json 
│       ├── test.json
│       ├── train.json
│       ├── features  # human feature dic
│       │   ├── dic  # keyward dict
│       │   │   ├── all_dic.json  # all the keyward
│       │   │   ├── get_train_dic.py
│       │   │   ├── thu_caijing_dic.json  # keyward From Tsinghua University
│       │   │   └── train_dic.json  
│       │   └── word_feature
│       │       ├── dev_flag_features.json
│       │       ├── dev_word_emb_features.json
│       │       ├── dev_word_features.json
│       │       ├── flag2id.json
│       │       └── ...
│       ├── get_mlm_data.py
│       ├── mlm  # pretrain the PLM
│       │   ├── mlm_dev.txt
│       │   └── mlm_train.txt
│       └──enhanced_train.json  # the augmented samples file
├── enhancement
│   └── replace.py  # replace keywords
├── mlm
│   ├── pretrain.sh
│   └── run_mlm.py
├── models
│   ├── GlobalPointer.py
│   ├── __init__.py
│   └── __pycache__
│       ├── GlobalPointer.cpython-36.pyc
│       └── __init__.cpython-36.pyc
├── src
│   ├── __pycache__
│   │   └── predict_CME.cpython-36.pyc
│   ├── ensemble.py
│   ├── predict_CME.py
│   └── train_CME.py
├── utils
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-36.pyc
│   │   ├── bert_optimization.cpython-36.pyc
│   │   ├── data_loader.cpython-36.pyc
│   │   ├── finetuning_argparse.cpython-36.pyc
│   │   ├── logger.cpython-36.pyc
│   │   └── ths_data_utills.cpython-36.pyc
│   ├── bert_optimization.py
│   ├── data_loader.py
│   ├── finetuning_argparse.py  # args settig files
│   ├── logger.py
│   └── ths_data_utills.py
└── word2vec
    ├── data
    │   └── word2vec_dic.json  # w2v location
    ├── gensim_ttl.py
    └── save_w2v_features.py
```

# My Modifications for Training
1. Hierarchical learning rate
2. output threshold
3. add a LSTM Layer
4. Add three artificial features
5. Continuing Pre training
6. R-drop
7. fgm : confrontation training
8. data augmentation：keyword replace
9. model ensemble
10. SWA
11. Boundary Smoothing

# Run
## data format
```
[
  [
    "和成都海光微相关的上市公司有哪些",
    [
      "成都海光微",
      "上市公司"
    ]
  ],
  [
    "股价异常波动停牌的股票",
    [
      "股价异常波动",
      "股票"
    ]
  ],
   ...
]
```


## Training
1. Download the pytorch pre-training model, the address is passed to the `--bert_model_path` parameter.
2. To start training, refer to `GP_runner.sh` for parameters or scripts, and `finetuning_argparse.py` for fine-tuning some parameters.

## artificial features
`in_dic`: Co-occurrence features, subscript set to 1 if `labeled keyword` is in the input text.\
`w2v_emb`: Splice `word_emb` to `token_emb`.\
`flag_id`: Add part-of-speech one-hot features.

### in_dic
Running conditions: `datasets/split_data/features/dic/get_train_dic.py` uses the `keyword` list `all_dic.json`, and it can be run.


### w2v_emb
Operating conditions:
1. Train yourself or find w2v features from the Internet, process them into a dictionary form `{ token : emb_list }`, and save them to `word2vec/data/word2vec_dic.json`.
2. Use `word2vec/save_w2v_features.py` to save the w2v features corresponding to each word to `datasets/split_data/features/word_feature/..._word_emb_features.json`.
3. Add the args parameter and run.

refer to in_dic

### flag_id
Refer to w2v_emb


## Data Augmentation：keyword Replace
1. Download or build a keyword list, here use THU-caijing corpus, put it in `datasets/split_data/features/dic/thu_caijing_dic.json`.
2. Run `enhancement/replace.py` to get the enhanced sample `datasets/split_data/enhanced_train.json`.

## Boundary Smoothing
1. `dataloader.py` 中加入 `self.get_boundary_smoothing()` Get a new soft label, compared with the source code, make the following changes:
    1. Modify dimension order for this project
    2. `[cls]` and `[sep]` cannot be inside soft label
    3. Allow `start index == end index`, that is, allow a single token as an entity
2. `train_CMR.py` - `multilabel_categorical_crossentropy()` Adjust the logic of calculating loss.

## Model ensembling
`ensemble.sh` 脚本中的 `checkpoints` 指定 checkpoint 列表，空格隔开。
每次训练后会在checkpoints中得到`ths_model.pth`模型参数文件和`args.json`args文件，可以手动将这两个文件剪贴到一个模型文件夹中，融合时，传入文件夹名称即可。
