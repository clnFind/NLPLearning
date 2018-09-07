# -*- coding:utf-8 -*-


from pyhanlp import *
from nlp.hanlp_test1 import ensure_data

WordVectorModel = JClass('com.hankcs.hanlp.mining.word2vec.WordVectorModel')
DocVectorModel = JClass('com.hankcs.hanlp.mining.word2vec.DocVectorModel')
model_path = os.path.join(
    ensure_data('hanlp-wiki-vec-zh', 'http://hanlp.linrunsoft.com/release/model/hanlp-wiki-vec-zh.zip'),
    'hanlp-wiki-vec-zh.txt')
word2vec = WordVectorModel(model_path)
doc2vec = DocVectorModel(word2vec)
docs = ["山东苹果丰收", "农民在江苏种水稻", "奥运会女排夺冠", "世界锦标赛胜出", "中国足球失败"]
for idx, doc in enumerate(docs):
    doc2vec.addDocument(idx, doc)

print(word2vec.nearest('语言'))

for res in doc2vec.nearest('我要看比赛'):
    print('%s = %.2f' % (docs[res.getKey().intValue()], res.getValue().floatValue()))
