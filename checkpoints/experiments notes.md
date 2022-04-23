## Baseline
```
    --bert_model_path ../pre_ckpts/bert-base-chinese \
    --file_path datasets/split_data \
    --batch_size 32 \
    --epoch 16
    --learning_rate 2e-5
```
best f1 scaore is 0.8370 (12)

## Roberta
```
    --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
    --file_path datasets/split_data \
    --batch_size 32 \
    --epoch 16 \
    --do_predict
    --learning_rate 2e-5
```
best f1 scaore is 0.8340 (10)
```
    --do_predict \
    --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
    --file_path datasets/split_data \
    --batch_size 64 \
    --epoch 16 \
    --learning_rate 2e-5
```
best f1 scaore is 0.8375 in epoch 11
test f1: 0.8373	    recall 0.8508	acc 0.8241
```
    --do_predict \
    --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
    --file_path datasets/split_data \
    --batch_size 16 \
    --epoch 16 \
    --learning_rate 2e-5
```
best f1 scaore is 0.8393442140269085 in epoch 15
test f1:  0.8396    recall 0.8502	acc 0.8291

### 尝试改变整体学习率
```
    --do_predict \
    --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
    --file_path datasets/split_data \
    --batch_size 16 \
    --epoch 16 \
    --learning_rate 5e-5
```
best f1 scaore is 0.8288165213908197 in epoch 12

```
    --do_predict \
    --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
    --file_path datasets/split_data \
    --batch_size 32 \
    --epoch 16 \
    --learning_rate 5e-5
```
best f1 scaore is 0.8348392547196896 in epoch 15

```
    --do_predict \
    --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
    --file_path datasets/split_data \
    --batch_size 64 \
    --epoch 16 \
    --learning_rate 5e-5
```
best f1 scaore is 0.8317408004913398 in epoch 13

### 尝试分层学习率
```
  --do_predict \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --file_path datasets/split_data \
  --batch_size 64 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 2e-4
```
best f1 scaore is 0.8421757271080833 in epoch 10
test f1: 0.8328     recall: 0.8439	acc: 0.822
```
  --do_predict \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --file_path datasets/split_data \
  --batch_size 32 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 2e-4
```
best f1 scaore is 0.8373177080987221 in epoch 10

```
  --do_predict \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --file_path datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 2e-4
```
best f1 scaore is 0.840136257005954 in epoch 7
test f1: 0.8454    recall: 0.8445	acc: 0.8464

### 修改encoder学习率
```
  --do_train \
  --do_predict \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --file_path datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 5e-5 \
  --biaffine_learning_rate 2e-4
```
 best f1 scaore is 0.8381020271890904 in epoch 15
 train f1: 0.8335    recall: 0.841	acc: 0.8262
 
```
  --do_train \
  --do_predict \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --file_path datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 1e-5 \
  --biaffine_learning_rate 2e-4
```
best f1 scaore is 0.8416437385221925 in epoch 8
test f1: 0.8358    recall: 0.8312	acc: 0.8404

```
  --do_train \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --file_path datasets/split_data \
  --batch_size 64 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 4e-4
```
best f1 scaore is 0.843537826882508 in epoch 15
0.8455	0.8606	0.830

```
  --do_train \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --file_path datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 4e-4
```
best f1 scaore is 0.8400366788635182 in epoch 14
0.8459	0.8537	0.8382

- 调整epoch=32，从结果看确实是没有训练充分
best f1 scaore is 0.8402776649101805 in epoch 21
0.8464	0.8635	0.83

bert+ drop +ffn: best f1 scaore is 0.841668187165405 in epoch 10
0.8436  0.8393  0.8481

## 增加特征
参数`features` : 
    - "in_dic"
    - "w2v_emb"
    - "flag_id"
    
### 字符是否在字典中，1/0
字典选择：（直接在代码中修改）
- all_dic.json : train和eval的phase
- train_dic.json :  只有train的phase
- thu_caijing_dic.json : Thuocl 清华共享的财经字典

代码修改：
1. `this_data_utils.py`中`read_example_form_file`函数
    1. 存储添加的特征维度 放入 `args.total_feature_size`
    2. 存储特征名称 `feature_names`
    3. `example` 字典中加入键`dic_features` 存储句子中的单词是否在字典中 eg. `[0,1,1,0,0,1,1,0]`
2. `data_loader.py`
    1. `EntDataset.encoder()` 方法将 `dic_features`和`subtoken`对其，得到 `dic_ids`
    2. `EntDataset.collate()` 方法给 `dic_ids` 做 padding ，得到 `batch_hand_features`
3. `GlobalPointer.py` 中
    1. `__init__()` 方法使用上面的 `args.total_feature_size` 初始化 `dense` 层
    2. `forward()` 方法使用上面 `batch_hand_features` 做传播
4. `predict_CME.py` 中，修改 `GlobalPointer` 模型初始化，并处理 `dic_ids` 放入模型。

baseline
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 4e-4 \
  --features in_dic
```
INFO: best f1 scaore is 0.8444643734273989 in epoch 3
0.8456	0.8612	0.8305

```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 32 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 4e-4 \
  --features in_dic
```
best f1 scaore is 0.8458423183916706 in epoch 13
0.8387 rec 0.8491	acc 0.8285

```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 64 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 4e-4 \
  --features in_dic
```
best f1 scaore is 0.8387467260407774 in epoch 15
0.8401	0.8549	0.8257

- 这里使用清华金融语料进行判定
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --feature in_dic
```


### 每个字添加w2v embedding
```
  --do_train \
  --do_predict \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 64 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 4e-4 \
  --features in_dic,w2v_emb
```
best f1 scaore is 0.8372482348337258 in epoch 6

```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 32 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 4e-4 \
  --features in_dic,w2v_emb
```
0.8404825182767469 in epoch 15
0.8378	rec: 0.845	 acc: 0.8306


```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 4e-4 \
  --features in_dic,w2v_emb
```
best f1 scaore is 0.8427898016049872 in epoch 9
0.8385	rec: 0.8404	 acc: 0.8365

#### 效果为什么不好呢：
1. 参数增多，尝试调大学习率
2. 之前两个 ffn 层是 768 -> 64，大概压缩到 10%
   现在是 768+301 = 1069 ，可能压缩太多了，尝试 64 -> 128

------
1. 调大学习率 biaffine_learning_rate 4e-4 --> 5e-4
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 5e-4 \
  --features in_dic,w2v_emb
```
best f1 scaore is 0.8433650576726767 in epoch 12
0.844	0.8601	0.8285
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 64 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --biaffine_learning_rate 5e-4 \
  --features in_dic,w2v_emb
```
best f1 scaore is 0.8395568486336432 in epoch 7
0.8389	0.8312	0.8468
确实能涨点，差不多0.2


------
2. 尝试 64 -> 128
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 5e-4 \
  --features in_dic,w2v_emb \
  --inner_dim 128
```
best f1 scaore is 0.8406495457259398 in epoch 6
又慢又拉.....

## 数据增强

```
[   ['机构评级最高的股票是', ['股票', '机构']], 
    ['光脚大阴线是什么样子', ['光脚大阴线']]   ]

变成

[   ['德明利评级最高的股市泡沫是', ['股市泡沫', '德明利']], 
    ['公告日期是什么样子', ['公告日期']]     ]
```
变更的word是字典中的word，可能有过拟合和数据泄露的风险，anyway，先实验吧。

-----
原本是： best f1 scaore is 0.8400366788635182 in epoch 14
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --do_enhance
```
- 替换train+eval（数据泄露）
best f1 scaore is 0.8491481834534141 in epoch 11
0.8285	0.8341	0.8231

- 可能数据泄露，只替换train中的字符。
best f1 scaore is 0.8330148776675512 in epoch 5
0.8204	0.8346	0.8067

## 使用 flag 信息
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --feature flag_id
```
best f1 scaore is 0.8438768748682931 in epoch 15
0.8445	0.856	0.8334


这里15达到最佳，猜测后面还能涨点，不然就过拟合，试一试 `epoch=20`
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 20 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --feature flag_id
```
best f1 scaore is 0.8482354551134914 in epoch 18
0.8336	0.8427	0.8246


- 调大学习率
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 5e-4 \
  --feature flag_id
```
 best f1 scaore is 0.8391340645426559 in epoch 12
 
## 使用 lstm
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --use_lstm
```
best f1 scaore is 0.8379270118470535 in epoch 12

- bert后加dropout，见前面 


```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 32 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --use_lstm
```
best f1 scaore is 0.8428336212330081 in epoch 15
0.8443	0.856	0.8329

```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 64 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --use_lstm
```
best f1 scaore is 0.8389056972303817 in epoch 13


- 调大学习率
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 32 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 5e-4 \
  --use_lstm
```
best f1 scaore is 0.8331771830239175 in epoch 9


- 调小学习率
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 32 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 3e-4 \
  --use_lstm
```
best f1 scaore is 0.8430570718425898 in epoch 11
0.8384	0.8387	0.8382


## 加了一个 bar
- baseline 0.01
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 64 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --output_bar 0.01
```
best f1 scaore is 0.8440785542055955 in epoch 8
0.8319	0.8398	0.8241


```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --output_bar 0.01
```
best f1 scaore is 0.841668187165405 in epoch 10
0.8436	0.8393	0.8481

- 尝试提高 bar
```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 64 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --output_bar 0.05
```
best f1 scaore is 0.8432484361712046 in epoch 8
0.8316	0.8393	0.824

## 聚合实验
```
  --do_train \
  --do_predict \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --feature in_dic,w2v_emb \
  --use_lstm
```
best f1 scaore is 0.8340290054646788 in epoch 14

```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 5e-4 \
  --feature in_dic,w2v_emb,flag_id
```
best f1 scaore is 0.8419296737143037 in epoch 8
0.8431	0.8531	0.8334


  
  
 

## r-dropout
```
  --do_train \
  --do_predict \
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --do_rdrop
```
默认 alpha = 4
best f1 scaore is 0.8284838901097161 in epoch 15

```
  --bert_model_path ../pre_ckpts/RoBERTa_zh_Large_PyTorch \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --do_rdrop \
  --rdrop_alpha 1
```
best f1 scaore is 0.8415481958106512 in epoch 6
0.8469	0.8589	0.8353   终于涨点了(qiao)

将bert后面的 dropout撤掉，果然还是不加drop了。
best f1 scaore is 0.8404404372300573 in epoch 10
0.8475	0.8722	0.8241


## 使用fgm
效果下降一个点


## 使用 mengzi-bert-base-fin
```
  --bert_model_path ../pre_ckpts/mengzi-bert-base-fin \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 16 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4
```
best f1 scaore is 0.8432644045240459 in epoch 12
0.8363	0.8404	0.8322

### 没有训练充分啊，多训练几次
best f1 scaore is 0.8456404115084426 in epoch 19
0.8428	0.8497	0.8361

```
  --bert_model_path ../pre_ckpts/mengzi-bert-base-fin \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 32 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --do_rdrop \
  --rdrop_alpha 1
```
best f1 scaore is 0.8519790292696489 in epoch 21
0.8516	0.8774	0.8272

- 调 batch_size
```
  --bert_model_path ../pre_ckpts/mengzi-bert-base-fin \
  --data_dir datasets/split_data \
  --batch_size 32 \
  --epoch 32 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --do_rdrop \
  --rdrop_alpha 1
```
best f1 scaore is 0.8477547823167991 in epoch 24
0.8573	0.8774	0.8382

- 修改 rdrop_alpha=0.5 （变小后效果下降了）
best f1 scaore is 0.8509213982949158 in epoch 27
0.8534	0.8682	0.8391

```
  --bert_model_path ../pre_ckpts/mengzi-bert-base-fin \
  --data_dir datasets/split_data \
  --batch_size 64 \
  --epoch 32 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --do_rdrop \
  --rdrop_alpha 1
```
best f1 scaore is 0.8504581792261222 in epoch 28
0.8573 0.8786 0.837

- 修改 rdrop_alpha=2 （变大）
f1 scaore is 0.8410156411885708 in epoch 31
 
- 修改 rdrop_alpha=0.5 （变小）
best f1 scaore is 0.8487486786697419 in epoch 25

## 预训练
- 刚开始效果不好，因为只使用了train+eval的语料。跑了三万步，效果下降。

### 现在修改为 train+eval+test 所有语料，并且抽取 5% 作为eval。
```
--batch_size 64 \
--epoch 32 \
--encoder_learning_rate 2e-5 \
--decoder_learning_rate 4e-4 \
--do_rdrop \
--rdrop_alpha 1
```

|epoch|eval f1|test f1|test recall|test pre|备注|
|---|---|---|---|---|---|
|3|0.8513|0.857|0.8803|0.8349||
|4|**0.8544**|0.8576|0.8739|0.8418|测试成绩最好|
|5|0.8530|0.8585|0.8768|0.8409||
|6|0.8531|0.863|0.8849|0.8421||
|6 !r-drop|0.8475|||||
|6_batch32|0.8513|0.8563|0.8716|0.8415|
|6_swa|0.8528|0.8627|0.8832|0.8432|自动情况下，保留的是后面几轮的checkpoint。应该用手动挡位，根据eval结果来记录。|
|7|0.8509|**0.8635**|0.8832|0.8446|线上成绩最好|
|7 !r-drop|0.8477|||||
|7_batch32|0.8508|0.8596|0.8745|0.8452|三个配置一起对比，pt这个最好。|
|7_fgm0.3|0.8496|||||
|8|0.8470| | | ||
|8 !r-drop|0.8505|||||
|9|0.8519|0.8648|0.8861|0.8446||
|9_16|0.8533|0.8561|0.8774|0.8359||
|10|0.8515|0.8658|0.8843|0.8481||


  
		
```  
--bert_model_path ../pre_ckpts/my_mengzi_7 \
  --data_dir datasets/split_data \
  --batch_size 64 \
  --epoch 32 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --do_rdrop \
  --rdrop_alpha 1 \
  --do_fgm \
  --fgm_epsilon 0.3  

```


best f1 scaore is 0.8496362957072521 in epoch 21

 

## 模型融合
```
checkpoints/pt6_64
checkpoints/pt7_64
```
0.8661	0.8867	0.8465


```
checkpoints/pt6_64
checkpoints/pt7_64
checkpoints/pt9_64
checkpoints/pt9_16
checkpoints/pt10_64\
```
0.8645 0.882 0.8477

```
checkpoints/pt9_64
checkpoints/pt10_64\
```
0.8644	0.8849	0.8449


```
checkpoints/pt9_64
checkpoints/pt10_64
checkpoints/pt6_64
checkpoints/pt7_64\
```
0.8673 0.8878 0.8476