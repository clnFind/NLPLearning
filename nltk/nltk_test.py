# -*- coding: utf-8 -*-

import nltk
from pprint import pprint
# 古腾堡语料库
from nltk.corpus import gutenberg
# 网络语料库
from nltk.corpus import webtext
# 读取文本内容
from nltk.corpus import PlaintextCorpusReader
# 就职演说语料库
from nltk.corpus import inaugural
# 即时消息聊天会话语料库
from nltk.corpus import nps_chat
# 布朗语料库，包含500个不同来源
from nltk.corpus import brown
# 路透社语料库
from nltk.corpus import reuters


def corpus():
    # 获取语料库中的所有文本文件
    l = gutenberg.fileids()
    print(l)

    # 获取文本的所有单词列表
    w = gutenberg.words("austen-emma.txt")
    print(w)

    # 获取语料
    names_corpus = PlaintextCorpusReader('../testdata', ['female.txt', 'male.txt'])
    all_names = names_corpus.words()
    print(all_names)

    # 返回文本原始字符串
    # raw_str = gutenberg.raw("austen-emma.txt")
    # print(raw_str)

    # 返回文本的句子列表，每个句子是单词列表
    sentences = gutenberg.sents("austen-emma.txt")
    print(sentences)

    # 获取网络文本语料库
    net_corpus = webtext.fileids()
    print(net_corpus, type(net_corpus))

    # 美总统就职演说语料
    sp = inaugural.fileids()
    print(sp[-10:])
    print(type(sp))
    print(sp[-6:-1])

    # 聊天会话语料
    tt = nps_chat.fileids()
    pprint(tt)

    # 返回对话列表，每个对话是单词列表;  05年10月19，30多岁，705个帖子
    chat_room = nps_chat.posts('10-19-30s_705posts.xml')
    print(chat_room[-3:])

    # 获取语料库里的所有类别
    cc = brown.categories()
    print(cc)
    c = brown.fileids(['news', 'lore'])
    print(c)
    print(brown.words('ca01'))

    # 路透社语料库的类别
    rr = reuters.categories()
    print(rr)
    print(len(rr))

# 条件概率  类别为条件， 单词为事件
pairs = [(genre, word) for genre in brown.categories() for word in brown.words(categories=genre)]
print(pairs[:4])
cfd = nltk.ConditionalFreqDist(pairs)
print(cfd.conditions())

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'him']
# 以表格形式显示
cfd.tabulate(conditions=genres, samples=modals)
# 图表
# cfd.plot(conditions=genres, samples=modals)

text = brown.words(categories='news')
# 双连词
bigrams_words = nltk.bigrams(text)
# print(list(bigrams_words)[:4])
cfd = nltk.ConditionalFreqDist(bigrams_words)
# print(cfd.conditions())
fd = cfd['can']
# 分析 can 后面的词
fd.plot(10)