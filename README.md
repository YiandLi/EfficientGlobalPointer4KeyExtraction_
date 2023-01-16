English Version READDME: [README_English.md](./README_English.md)


# 模型源码
原模型地址：https://github.com/xhw205/Efficient-GlobalPointer-torch

基于 GlobalPointer 的改进，[Keras 版本](https://spaces.ac.cn/archives/8877) 的 torch 复现，核心还是 token-pair 。
绝大部分代码源自作者之前关于 GlobalPointer 的 [repository](https://github.com/xhw205/GlobalPointer_torch)。

# 更新记录
- 2022/04/23 创建
- 2022/6/23 增加 boundary smoothing 功能

# 依赖
本地测试时，使用CPU环境，mlm以外均能跑通，依赖为
```
python==3.6
torch==1.8.1
transformers==4.4.1
```

真实训练时使用服务器，可以跑通mlm，依赖为：
```
python==3.7
torch==1.8.1
transformers==4.10.0
```
服务器所有依赖参考`server_requirements.txt`，千万不要让pycharm默认全部安装。

# 文件
```
EfficientGlobalPointer4KeyExtraction
├── ensemble.sh  # 模型融合脚本
├── GP_runner.sh  # finetune脚本
├── README.md
├── server_requirements.txt  # 服务器依赖
├── result.txt  # 预测结果
├── err.log
├── checkpoints  # 模型保存地址
│   ├── bad_cases_GP.txt # badcase 输出
│   └── experiments_notes.md  # 实验记录
├── datasets
│   └── split_data
│       ├── biaffine_labels.txt  # 实体标签文件（实体类型，不加BIO）
│       ├── dev.json  # finetune数据
│       ├── test.json
│       ├── train.json
│       ├── features  # 人工特征文件夹
│       │   ├── dic  # keyward 字典
│       │   │   ├── all_dic.json  # 所有的keyward
│       │   │   ├── get_train_dic.py
│       │   │   ├── thu_caijing_dic.json  # 网上搜集语料的keyward
│       │   │   └── train_dic.json  # 训练集中的keyward
│       │   └── word_feature
│       │       ├── dev_flag_features.json
│       │       ├── dev_word_emb_features.json
│       │       ├── dev_word_features.json
│       │       ├── flag2id.json
│       │       └── ...
│       ├── get_mlm_data.py
│       ├── mlm  # 预训练数据
│       │   ├── mlm_dev.txt
│       │   └── mlm_train.txt
│       └──enhanced_train.json  # 实体替换输出的增强样本
├── enhancement
│   └── replace.py  # 数据增强：替换keyward
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
│   ├── finetuning_argparse.py  # args 参数设置文件
│   ├── logger.py
│   └── ths_data_utills.py
└── word2vec
    ├── data
    │   └── word2vec_dic.json  # w2v 地址
    ├── gensim_ttl.py
    └── save_w2v_features.py
```

# My Modifications for Training
1. 分层学习率
2. 输出阈值
3. 加入 LSTM
4. 加入三个人工特征
5. 继续预训练 
6. R-drop
7. fgm 对抗训练
8. 数据增强：keyword 替换
9. 模型融合
10. SWA
11. Boundary Smoothing

# 运行
## 数据集格式
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
嵌套列表，输入 text 和 keyword 列表。

## 训练
1. 下载 pytorch 预训练模型，地址传入 `--bert_model_path` 参数。
2. 开始训练，参数或脚本参考 `GP_runner.sh`，微调部分参数参考 `finetuning_argparse.py`。 

## 人工特征
`in_dic`: 共现特征，如果 `labeled keyword` 在输入文本中，则下标设置为 1。
`w2v_emb`: 将`word_emb`拼接到`token_emb`。
`flag_id`: 加入词性 one-hot 特征。

### in_dic
运行条件：`datasets/split_data/features/dic/get_train_dic.py` 使用得到 `keyword` 列表 `all_dic.json`，即可运行。

代码实现
1. `this_data_utils.read_example_form_file`函数
    1. 存储添加的特征维度 放入 `args.total_feature_size`。
    2. `example` 字典中加入键`dic_features` 存储句子中的单词是否在字典中 eg. `[0,1,1,0,0,1,1,0]`
2. `data_loader.py`
    1. `EntDataset.encoder()` 方法将 `dic_embs`填充到`feature_embs`中，并return。
    2. `EntDataset.collate()` 方法给 `feature_embs` 做 padding ，得到 `batch_handfeatures`，并return。
3. `GlobalPointer.py` 中
    1. `__init__()` 方法使用上面的 `feature_size` 初始化 `dense` 层
    2. `forward()` 方法使用上面 `feature_size` 做传播
4. `predict_CME.py` 中，修改 `GlobalPointer` 模型初始化，并处理 `dic_ids` 放入模型。

### w2v_emb
运行条件：
1. 自行训练或从网上找到w2v特征，处理成字典形式 `{ token : emb_list }` 形式，保存到 `word2vec/data/word2vec_dic.json`。
2. 使用 `word2vec/save_w2v_features.py` 保存每个字对应的 w2v 特征到 `datasets/split_data/features/word_feature/..._word_emb_features.json` 。
3. 加入args参数，运行。

代码实现：参考 in_dic

### flag_id
参考 w2v_emb


## 数据增强：keyword替换
1. 下载或构建关键词列表，这里使用 THU-caijing 语料，放在`datasets/split_data/features/dic/thu_caijing_dic.json` 中。
2. 运行 `enhancement/replace.py` ，得到增强样本 `datasets/split_data/enhanced_train.json` 。

## Boundary Smoothing
1. `dataloader.py` 中加入 `self.get_boundary_smoothing()` 得到新的 soft label，和源码相比，做出如下改动：
    1. 针对本项目修改维度顺序
    2. `[cls]` 和 `[sep]` 不能在 soft label 内
    3. 允许 `start index == end index`，即允许单个 token 作为实体
2. `train_CMR.py` 中 `multilabel_categorical_crossentropy()` 调整计算损失的逻辑。

## 模型融合
`ensemble.sh` 脚本中的 `checkpoints` 指定 checkpoint 列表，空格隔开。
每次训练后会在checkpoints中得到`ths_model.pth`模型参数文件和`args.json`args文件，可以手动将这两个文件剪贴到一个模型文件夹中，融合时，传入文件夹名称即可。
