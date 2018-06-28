# -*- coding:utf-8 -*-

import fasttext
import jieba


def text_to_line(text_path):
    """
    文本去停用词
    返回 （一行）字符串文本信息
    :param text_path:
    :return:
    """
    stop_wordspath = './CNENstopwords.txt'
    stopwords = {}.fromkeys([line.rstrip() for line in open(stop_wordspath, "r", encoding='utf-8')])
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 结巴分词
    seg_text = jieba.cut(text.replace("\t", " ").replace("\n", " "))
    outline = " ".join(seg_text)
    outline = " ".join(outline.split())

    # 去停用词
    outline_list = outline.split()
    outline_list_filter = [item for item in outline_list if item not in stopwords]
    outline = " ".join(outline_list_filter)

    return outline


if __name__ == '__main__':

    # model_path = './data/thucnews.dat.seg.shuf.model.bin'
    model_path = './data/sogounews.dat.seg.shuf.model.bin'

    classifier = fasttext.load_model(model_path, label_prefix='__label__')

    # file_path = "./test/军事/0007.txt"
    file_path = "./test/体育/0004.txt"
    files_str = text_to_line(file_path)
    print(files_str)

    texts = [files_str]
    labels = classifier.predict(texts)
    print(labels)

    # Or with the probability
    labels = classifier.predict_proba(texts)
    print(labels)

    labels = classifier.predict(texts, k=3)
    print(labels)

    # Or with the probability
    labels = classifier.predict_proba(texts, k=3)
    print(labels)
