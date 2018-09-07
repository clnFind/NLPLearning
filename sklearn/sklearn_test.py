# -*- coding: utf-8 -*-

import jieba
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.cluster import KMeans


def train_corpus(dataset):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(dataset)
    km = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=1, verbose=False)

    km.fit(X)
    # 保存模型
    joblib.dump(km,  './data/news_cluster_test.pkl')


def jieba_tokenize(text):
    return jieba.cut(text)

tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize, lowercase=False)
'''
tokenizer: 指定分词函数
lowercase: 在分词之前将所有的文本转换成小写，因为涉及到中文文本处理，
所以最好是False
'''
text_list = ["今天天气真好啊啊啊啊", "小明上了清华大学", "我今天拿到了Google的Offer", "清华大学在自然语言处理方面真厉害"]


hv = HashingVectorizer(tokenizer=jieba_tokenize, n_features=10)
tt = hv.transform(text_list)


print(tt)

# 需要进行聚类的文本集
tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)

print(tfidf_vectorizer)

# print(tfidf_matrix)

terms = tfidf_vectorizer.get_feature_names()

print(terms)
print(len(terms))

num_clusters = 2
km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=1, init='k-means++', n_jobs=1)
"""
n_clusters: 指定K的值
max_iter: 对于单次初始值计算的最大迭代次数
n_init: 重新选择初始值的次数
init: 制定初始值选择的算法
n_jobs: 进程个数，为-1的时候是指默认跑满CPU
注意，这个对于单个初始值的计算始终只会使用单进程计算，
并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40, 
服务器上面有20个CPU可以开40个进程，最终只会开10个进程
"""

# 返回各文本的所被分配到的类索引
result = km_cluster.fit_predict(tfidf_matrix)

rst = list(km_cluster.predict(tfidf_matrix))
print('Cluster distribution:')
print(rst)

print([(i, rst.count(i)) for i in rst])
print(dict([(i, rst.count(i)) for i in rst]))

cluster_centers = km_cluster.cluster_centers_
print("##### ", cluster_centers)

clusters = km_cluster.labels_

print(clusters)
print("Predicting result: ", result)
for i in range(2):
    print("cluster %s: " % i, end='')
    # 每类取5个特征
    for ind in cluster_centers.argsort()[i, :5]:
        print(' %s' % terms[ind], end='')
    print()

print(cluster_centers)
print(cluster_centers.argsort()[:, ::-1])
print(cluster_centers.argsort())

# # 保存模型和加载
# joblib.dump(km_cluster,  './data/news_cluster.pkl')
# km = joblib.load('./data/news_cluster.pkl')
# clusters = km.labels_
