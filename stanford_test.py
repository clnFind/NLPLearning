# -*- coding: utf-8 -*-

from nltk.tokenize.stanford_segmenter import StanfordSegmenter

segmenter = StanfordSegmenter(
    path_to_jar=r"/Users/cln/stanford-corenlp/segmenter/stanford-segmenter-3.9.1.jar",
    path_to_slf4j=r"/Users/cln/stanford-corenlp/slf4j-api.jar",
    java_class=r"edu.stanford.nlp.ie.crf.CRFClassifier",        # 分词模型
    path_to_model=r"/Users/cln/stanford-corenlp/segmenter/data/pku.gz",
    path_to_dict=r"/Users/cln/stanford-corenlp/segmenter/data/dict-chris6.ser.gz",
    path_to_sihan_corpora_dict=r"/Users/cln/stanford-corenlp/segmenter/data"
)

strs = "中国的一带一路带动了很多国家的发展。"

# 汉语分词
ch_result = segmenter.segment(strs)
print('汉语分词：\n', ch_result)
print(type(ch_result), '\n')


from nltk.tokenize import word_tokenize

sent = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
# 英语分词
rr = word_tokenize(sent)
print('英语分词：\n', rr, '\n')


from nltk.tag import StanfordNERTagger

# 英文命名实体识别
eng_tagger = StanfordNERTagger(r'/Users/cln/stanford-corenlp/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                               path_to_jar=r'/Users/cln/stanford-corenlp/stanford-ner/stanford-ner.jar')

eng_rst = eng_tagger.tag('Rami Eid is studying at Stony Brook University in NY'.split())

print('英语命名实体识别：\n', eng_rst, '\n')


# 汉语命名实体识别
ch_tagger = StanfordNERTagger(r'/Users/cln/stanford-corenlp/stanford-ner/classifiers/chinese.kbp.distsim.crf.ser.gz',
                              path_to_jar=r'/Users/cln/stanford-corenlp/stanford-ner/stanford-ner.jar')

texts = r"欧洲 东部 的 罗马尼亚 首都 是 布加勒斯特 也 是 一 座 世界性 的 城市 北京 南阳 普京 中国 习主席"
ch_rst = ch_tagger.tag(texts.split())

print('汉语命名实体识别：\n', ch_rst, '\n')


from nltk.tag import StanfordPOSTagger

# 汉语词性标注
chi_tagger = StanfordPOSTagger(r'/Users/cln/stanford-corenlp/postagger/models/chinese-distsim.tagger',
                              path_to_jar=r'/Users/cln/stanford-corenlp/postagger/stanford-postagger.jar')
print("汉语词性标注：")
print(chi_tagger.tag(ch_result.split()))

for _, word_and_tag in chi_tagger.tag(ch_result.split()):
    word, tag = word_and_tag.split('#')
    print(word, tag)

print('\n')

from nltk.parse.stanford import StanfordParser
from nltk import Tree
# 汉语句法分析
chi_parser = StanfordParser(r"/Users/cln/stanford-corenlp/parser/stanford-parser.jar",
                            r"/Users/cln/stanford-corenlp/parser/stanford-parser-3.9.1-models.jar",
                            r"/Users/cln/stanford-corenlp/models/lexparser/chinesePCFG.ser.gz")
rst = list(chi_parser.parse(ch_result.split()))
print('汉语句法分析：\n', rst, '\n')

tree = Tree.fromstring(rst)
tree.drow()

# print(rst[0].draw())


from nltk.parse.stanford import StanfordDependencyParser

# 汉语依存句法分析
dep_parser = StanfordDependencyParser(r"/Users/cln/stanford-corenlp/parser/stanford-parser.jar",
                                      r"/Users/cln/stanford-corenlp/parser/stanford-parser-3.9.1-models.jar",
                                      r"/Users/cln/stanford-corenlp/models/lexparser/chinesePCFG.ser.gz")
res = list(dep_parser.parse(ch_result.split()))
print("汉语依存句法分析：")
for row in res[0].triples():
    print(row)
