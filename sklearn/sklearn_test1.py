# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib
# import time
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from nlp.fasttext_identify import text_to_line


# from sklearn.decomposition import LatentDirichletAllocation


def loadDataset(corpus):
    """导入文本数据集"""
    dataset = []
    with open(corpus, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(line)
    return dataset


def transform(dataset, n_features=1000):
    # 5次以下的词汇忽略
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=5, use_idf=True)
    X = vectorizer.fit_transform(dataset)
    return X, vectorizer


def train(X, vectorizer, true_k=10, minibatch=False, showLable=False):

    # 使用采样数据还是原始数据训练k-means，
    if minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=False)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1,
                    verbose=False)
        # km = Birch(n_clusters=true_k)

    km.fit(X)
    # 保存模型
    joblib.dump(km,  './data/news_cluster.pkl')

    if os.path.exists('./data/news_cluster.pkl'):
        km = joblib.load('./data/news_cluster.pkl')
        if showLable:
            print("Top terms per cluster:")
            # 类别特征 逆序（概率从大到小）
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            # order_centroids = km.subcluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names()
            print(vectorizer.get_stop_words())
            for i in range(true_k):
                print("Cluster %d:" % i, end='')
                # 每类取10个特征
                for ind in order_centroids[i, :20]:
                    print(' %s' % terms[ind], end='')
                    # print(ind)
                print()

        # 预测每类的数量
        result = list(km.predict(X))
        print('Cluster distribution:')
        print(dict([(i, result.count(i)) for i in result]))
        return -km.score(X)
        # pass


def load_model(X, vectorizer, true_k=10):
    """
    加载训练好的模型
    :param X:
    :param vectorizer:
    :param true_k:
    :return:
    """

    if os.path.exists('./data/news_cluster.pkl'):
        km = joblib.load('./data/news_cluster.pkl')
        print("Top terms per cluster:")
        # 类别特征 逆序（概率从大到小）
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print(vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            # 每类取10个特征
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
                # print(ind)
            print()
        # 预测每类的数量
        result = list(km.predict(X))
        print(km.labels_)
        print('Cluster distribution:')
        print(dict([(i, result.count(i)) for i in result]))
        print(-km.score(X)/5000)


def test(corpus):
    """ 测试选择最优参数 """

    dataset = loadDataset(corpus)
    print("%d documents" % len(dataset))
    X, vectorizer = transform(dataset, n_features=300)
    true_ks = []
    scores = []
    for i in range(5, 20, 1):
        score = train(X, vectorizer, true_k=i)/len(dataset)
        print(i, score)
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(8, 4))
    plt.plot(true_ks, scores, label="error", color="red", linewidth=1)
    plt.xlabel("n_features")
    plt.ylabel("error")
    plt.legend()
    plt.show()


def show_result_pic(n_clusters, corpus):

    # dataset = loadDataset(corpus)
    # print("%d documents" % len(dataset))
    # X, vectorizer = transform(dataset, n_features=300)

    km = joblib.load('./data/news_cluster.pkl')
    print(km.labels_)
    cluster_centers = km.cluster_centers_

    print(cluster_centers[0, :])
    print(type(cluster_centers[0, :]), len(cluster_centers[0, :]))

    print(len(cluster_centers))

    markers = ['^', 'x', 'o', '*', '+']
    matplotlib.rcParams['font.sans-serif'] = 'SimHei'
    for i in range(n_clusters):
        members = km.labels_ == i
        # print(members)
      # for j in cluster_centers[members]:
        plt.scatter(cluster_centers[i], cluster_centers[i], s=60, marker=markers[i], c='b', alpha=0.5)
    plt.title('新闻聚类图示')
    plt.show()


def identify_text(X):

    # 搜狗新闻模型
    km = joblib.load('./data/news_cluster.pkl')
    # km = joblib.load('./data/thucnews_cluster.pkl')

    # THUCNews
    # km = joblib.load('./data/thucnews_cluster.pkl')

    result = km.predict(X)
    print(result)


def out(corpus):
    """在最优参数下输出聚类结果"""
    dataset = loadDataset(corpus)
    print("%d documents" % len(dataset))
    X, vectorizer = transform(dataset, n_features=500)
    # 提取5类
    score = train(X, vectorizer, true_k=5, showLable=True)/len(dataset)
    print(score)

if __name__ == '__main__':
    # corpus = "./data/news.dat.seg"
    # dataset = loadDataset(corpus)
    # print(dataset)
    # print(len(dataset))

    corpus = "./data/sogounews.dat.seg.shuf"
    # corpus = "./data/thucnews.dat.seg.shuf"

    # t1 = time.time()
    # # test(corpus)
    #
    # out(corpus)
    # t2 = time.time()
    # print("耗时：", t2-t1)

    dataset = loadDataset(corpus)
    X, vectorizer = transform(dataset, n_features=500)
    load_model(X, vectorizer, true_k=5)

    file_path = "./test/体育/0004.txt"
    file_str = [text_to_line(file_path)]
    print("输入的文本分词处理后的结果：\n", file_str)
    # print(len(file_str[0]))
    # X = TfidfVectorizer(max_features=300).fit_transform(file_str)
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(file_str)
    tt = vectorizer.get_feature_names()
    # print(tt)
    # print(len(tt))
    print("预测结果属于那一个簇（即类别）：")
    identify_text(X)

    # show_result_pic(5, corpus)


