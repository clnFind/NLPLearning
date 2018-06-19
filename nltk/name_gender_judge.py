# -*- coding: utf-8 -*-
import nltk
import random
from nltk.classify import apply_features
from nltk.corpus import PlaintextCorpusReader

import pickle
import os


def name_features(name):
    """
    名称特征提取
    :param name: 名称
    :return: 名称特征
    """
    name_chs = set([ch.lower() for ch in name])
    features = {}
    for ch in ch_features:
        features['contain(%s)' % ch] = (ch in name_chs)
    return features


if __name__ == '__main__':

    # 加载数据，按空白分割
    names_corpus = PlaintextCorpusReader('../testdata', ['female.txt', 'male.txt'])
    all_names = names_corpus.words()

    print(all_names)
    print(type(all_names), type(list(all_names)))
    l = list(all_names)
    print(type(l), l)

    # print(all_names[0:2], all_names[-4:-1])
    # print(len(all_names))
    #
    ch_freq = nltk.FreqDist(ch.lower() for name in all_names for ch in name)
    ch_freq_most = ch_freq.most_common(1000)
    ch_features = [ch for (ch, count) in ch_freq_most]

    print(name_features("周恩来"))

    # 打印出现最频繁（次数最多）的1000个字
    print(len(ch_features), ch_features)
    #
    female_names = [(name, 'female') for name in names_corpus.words('female.txt')]
    male_names = [(name, 'male') for name in names_corpus.words('male.txt')]
    #
    total_names = female_names + male_names
    print(total_names)
    print(total_names[-100:])
    # # 打乱数据
    random.shuffle(total_names)

    # 取60%的数据训练
    train_set_size = int(len(total_names) * 0.6)
    train_names = total_names[:train_set_size]
    test_names = total_names[train_set_size:]

    train_set = apply_features(name_features, train_names, True)
    test_set = apply_features(name_features, test_names, True)

    model_file = "../testdata/my_classifier_model.pickle"
    if not os.path.isfile(model_file):

        # 贝叶斯分类器训练模型
        classifier = nltk.NaiveBayesClassifier.train(train_set)

        # 保存训练模型
        with open(model_file, 'wb') as f:
            pickle.dump(classifier, f)

        # 训练集和测试集上的正确率
        print(nltk.classify.accuracy(classifier, train_set))
        print(nltk.classify.accuracy(classifier, test_set))

        # 查看分析结果
        classifier.show_most_informative_features(20)

    else:
        f = open(model_file, 'rb')
        classifier = pickle.load(f)

        print(classifier.classify(name_features('贾静雯')))
        print(classifier.classify(name_features('王宝强')))
        print(classifier.classify(name_features('刘胡兰')))
        print(classifier.classify(name_features('毛泽东')))
        print(classifier.classify(name_features('周恩来')))
        print(classifier.classify(name_features('Jane')))
        print(classifier.classify(name_features('Tom')))

        f.close()
