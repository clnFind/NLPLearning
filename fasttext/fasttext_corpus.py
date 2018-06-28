# -*- coding: utf-8 -*-

import os
import jieba
import time


def fasttext_corpus_deal(basedir, save_file, stop_wordspath):
    """
    分词处理文本文件
    每个文本文件以 一行+label 的形式写入保存的文件
    :param basedir:
    :param save_file:
    :return:
    """
    stopwords = {}.fromkeys([line.rstrip() for line in open(stop_wordspath, "r", encoding='utf-8')])

    data = open(save_file, 'w', encoding='utf-8')
    pre_dirs = os.listdir(basedir)
    print(pre_dirs)
    for folder in pre_dirs:
        filepath = os.path.join(basedir, folder)
        files = os.listdir(filepath)

        for _file in files:
            deal_file = os.path.join(filepath, _file)
            with open(deal_file, 'r', encoding='utf-8') as f:
                text = f.read()
            # 结巴分词
            seg_text = jieba.cut(text.replace("\t", " ").replace("\n", " "))
            outline = " ".join(seg_text)
            outline = " ".join(outline.split())

            # 去停用词
            outline_list = outline.split()
            outline_list_filter = [item for item in outline_list if item not in stopwords]
            outline = " ".join(outline_list_filter)

            outline = outline + "\t__label__" + folder + "\n"
            data.write(outline)
            data.flush()

    data.close()


if __name__ == '__main__':
    # basedir = "./test"
    # save_file = "./data/news.dat.seg"

    basedir = "./data/搜狗文本分类语料库迷你版"
    save_file = "./data/sogounews.dat.seg"

    # 停用词表
    stop_wordspath = './CNENstopwords.txt'

    print("begin to deal news text........... ")
    t1 = time.time()
    fasttext_corpus_deal(basedir, save_file, stop_wordspath)
    t2 = time.time()

    print("新闻汉语语料文本处理完成，耗时：%s 秒" % (t2 - t1))
