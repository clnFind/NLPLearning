# -*- coding: utf-8 -*-

import fasttext
import time


def fasttext_train(train_corpus, model_path):

    t1 = time.time()
    # 训练数据和测试数据 要对应绝对路径
    classifier = fasttext.supervised(train_corpus, model_path, label_prefix="__label__", thread=4)
    t2 = time.time()
    print("新闻数据训练耗时：%s  秒" % (t2 - t1))
    return


def test():

    classifier = fasttext.load_model('/homei/data/thucnews.dat.seg.shuf.model.bin', encoding='utf-8')
    result = classifier.test("./data/sogounews.dat.seg.shuf.test")
    print("准确度：", result.precision)
    print("召回率：", result.recall)


if __name__ == '__main__':

    print("begin to deal news corpus ........... ")
    # train_corpus = "./data/sogounews.dat.seg.shuf.train"
    # model_path = "./data/sogounews.dat.seg.shuf.model"

    train_corpus = "./data/sogounews.dat.seg.shuf.train"
    model_path = "./data/sogounews.dat.seg.shuf.model"
    fasttext_train(train_corpus, model_path)

