import jieba
from gensim.models import Word2Vec
#  停用词
stopword_list = []

# 语料
content = [
    "长江是中国第一大河，干流全长6397公里（以沱沱河为源），一般称6300公里。流域总面积一百八十余万平方公里，年平均入海水量约九千六百余亿立方米。以干流长度和入海水量论，长江均居世界第三位。",
    "黄河，中国古代也称河，发源于中华人民共和国青海省巴颜喀拉山脉，流经青海、四川、甘肃、宁夏、内蒙古、陕西、山西、河南、山东9个省区，最后于山东省东营垦利县注入渤海。干流河道全长5464千米，仅次于长江，为中国第二长河。黄河还是世界第五长河。",
    "黄河,是中华民族的母亲河。作为中华文明的发祥地,维系炎黄子孙的血脉.是中华民族民族精神与民族情感的象征。",
    "黄河被称为中华文明的母亲河。公元前2000多年华夏族在黄河领域的中原地区形成、繁衍。",
    "在兰州的“黄河第一桥”内蒙古托克托县河口镇以上的黄河河段为黄河上游。",
    "黄河上游根据河道特性的不同，又可分为河源段、峡谷段和冲积平原三部分。 ",
    "黄河,是中华民族的母亲河。"
]

# 分词
seg = [jieba.lcut(text) for text in content]

# 清洗
content_clean = []
for t in seg:
    text_clean = []
    for i in t:
        if len(i) > 1 and i != '\t\n':
            if not i.isdigit():
                if i.strip() not in stopword_list:
                    text_clean.append(i.strip())
    content_clean.append(text_clean)

# 用gensim训练词向量模型
model = Word2Vec(content_clean, sg=1, vector_size=100, window=5, min_count=2, negative=1,
                 sample=0.001, workers=4)
'''
sg=1 是 skip-gram 算法，对低频词敏感；默认 sg=0 为 CBOW 算法。
size 是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
window 是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b 个词，后面看 b 个词（b 在0-3之间随机）。
min_count 是对词进行过滤，频率小于 min-count 的单词则会被忽视，默认值为5。
negative 和 sample 可根据训练结果进行微调，sample 表示更高频率的词被随机下采样到所设置的阈值，默认值为 1e-3。
hs=1 表示层级 softmax 将会被使用，默认 hs=0 且 negative 不为0，则负采样将会被选择使用。
'''

# 训练后的模型model可以保存，备用
model.save('./word2vec')  # 保存
model = Word2Vec.load('word2vec')  # 加载model

# 获取词汇
words = model.wv.index_to_key
print(words)  # 长度18

# 获取对应词向量
vectors = model.wv.vectors
print(vectors)  # 18*100   100为设置的size，即词向量维度

# 根据指定词获取该词的向量
vec = model.wv['长江']
print(vec)

# 判断词之间的相似度
print(model.wv.similarity('黄河', '黄河'))  # 1.0
print(model.wv.similarity('黄河', '长江'))  # -0.08
print(model.wv.similarity('黄河', '中国'))  # 0.14

# 预测与'黄河'和'母亲河'最相似，而与长江不接近的词
print(model.wv.most_similar(positive=['黄河', '母亲河'], negative=['长江']))
