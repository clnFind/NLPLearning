# -*- coding: utf-8 -*-
import matplotlib
import os
import nltk
from nltk.corpus import PlaintextCorpusReader
import time


def extract_keys(deal_dirs, save_path, numbers, save_file, flag):
    """
    批量提取文本中的出现频率最高的关键词
    :param deal_dirs: 处理文本的父目录
    :param save_path: 处理结果保存的路径
    :param numbers: 提取词数量
    :param save_file: 提取结果保存的文件
    :return:
    """

    news_corpus = PlaintextCorpusReader(deal_dirs, '.*')
    files = news_corpus.fileids()
    print(files)

    # 创建保存文件目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(save_path, ' 创建成功!')

    savepath = os.path.join(save_path, save_file)

    for file in files:
        deal_f = PlaintextCorpusReader(deal_dirs, ['{}'.format(file)])
        word_list = deal_f.words()

        fdist1 = nltk.FreqDist(word_list)
        result = fdist1.most_common(numbers)
        print(result)

        ss = ''
        for word, num in result:
            if flag:
                ss += ("%s: %s " % (word, num))
            ss += word + " "
        save_analy_result(savepath, ss+'\n')


def save_analy_result(path, ss):

    with open(path, "a", encoding='utf-8') as f:
        f.write(ss)


if __name__ == '__main__':

    deal_dirs = "/Users/cln/SogouCorpus/test/C000014"
    save_path = "/Users/cln/SogouCorpus/test"
    num = 10
    save_file = "analy_result.txt"
    # True 表示分析结果带有词频数， False 表示不显示词频数
    flag = False

    t1 = time.time()
    extract_keys(deal_dirs, save_path, num, save_file, flag)
    t2 = time.time()

    print("新闻汉语语料(300个)词频统计完成，耗时：%s 秒" % (t2 - t1))



# deal_dirs = "/Users/cln/SogouCorpus/test/C000014"
# news_corpus = PlaintextCorpusReader(deal_dirs, '.*')
# files = news_corpus.fileids()
# print(files)
#
#
# # for file in files:
# deal_f = PlaintextCorpusReader(deal_dirs, ['1.txt'])
#
# # 返回原始文本信息
# texts = deal_f.raw()
# print(texts)
#
# word_list = deal_f.words()
# print(word_list)
#
# fdist1 = nltk.FreqDist(word_list)
# print("次数最多的词：", fdist1.max(), '\n')
#
# print("出现一次的词(输出10个)：", fdist1.hapaxes()[:10], '\n')
#
# # 输出次数最多的前15个词
# print("次数最多的前15个：", fdist1.most_common(15), '\n')


