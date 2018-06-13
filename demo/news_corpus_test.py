# -*- coding: utf-8 -*-
import matplotlib

import nltk
# from nltk.corpus import PlaintextCorpusReader

# news_corpus = PlaintextCorpusReader('/Users/cln/SogouCorpus/FileNews', '.*')
# files = news_corpus.fileids()
#
# print(files)
#
# file_corpus = PlaintextCorpusReader('/Users/cln/SogouCorpus/FileNews', ['C000014/0.txt'])
#
# ww = file_corpus.words()
#
# print(ww)


import sys
import jieba.posseg as pseg

sys.path.append("../")
# jieba.load_userdict("../Database/userdict.txt") # 加载自定义分词词典


def cut_txt_word(deal_path, stopwords_path):
    """
    分词.词性标注以及去停用词
    stopwordspath： 停用词路径
    dealpath：中文数据预处理文件的路径
    """
    stopwords = {}.fromkeys([line.rstrip() for line in open(stopwords_path, "r", encoding='utf-8')])   # 停用词表
    # stopwords_l = [line.rstrip() for line in open(stopwords_path, "r", encoding='utf-8')]  # 比上面的耗时

    print(stopwords)
    with open(deal_path, "r", encoding='utf-8') as f:
        txtlist = f.read()              # 读取待处理的文本

    # print(txtlist)
    words = pseg.cut(txtlist)           # 带词性标注的分词结果

    cut_result = ""                      # 获取去除停用词后的分词结果
    for word, flag in words:

        if word not in stopwords and not word.isspace():
            cut_result += word + " "      # 去停用词

    return cut_result

if __name__ == '__main__':

    stopwords_path = "./CNENstopwords.txt"
    deal_path = "/Users/cln/SogouCorpus/Sample/C000007/7.txt"
    save_path = "./test_coupus.txt"

    cut_rst = cut_txt_word(deal_path, stopwords_path)
    print(cut_rst)

    tt = nltk.word_tokenize(cut_rst)
    print(tt, '\n')

    # 获取词频
    fdist1 = nltk.FreqDist(tt)

    print("次数最多的词：", fdist1.max(), '\n')

    print("出现一次的词(输出10个)：", fdist1.hapaxes()[:10], '\n')

    # 输出次数最多的前15个词
    print("次数最多的前15个：", fdist1.most_common(15), '\n')

    # 获取前10个词和词频
    for key, val in sorted(fdist1.items(), key=lambda x: (x[1], x[0]), reverse=True)[:10]:
        print(key, val)

    # 获取某个词的出现数量
    print(fdist1['集团'])
    # 设置字体，中文显示
    matplotlib.rcParams['font.sans-serif'] = 'SimHei'
    fdist1.plot(10, cumulative=False, title="词频统计图解")
