# -*- coding: utf-8 -*-

import jieba
import jieba.analyse as analyse
import jieba.posseg as pseg

# 全模式
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print(seg_list, type(seg_list))
print("Full Mode: " + "/ ".join(seg_list))

# 精确模式  cut_all 默认值为False
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))
#
# 默认是精确模式
seg_list = jieba.cut("他来到了网易杭研大厦")
print(", ".join(seg_list))
#
# 搜索引擎模式
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print(", ".join(seg_list))


# text_path = '../testdata/female.txt' #设置要分析的文本路径
# text = open("../testdata/female.txt", 'rb')
#
# tt = text.read()

tt = "习近平是中华人民共和国的国家主席，李克强是总理，习近平祖籍是河南邓州。"
# 使用jieba.analyse.extract_tags()参数提取关键字,默认参数为20， allowPos 默认过滤词性为空  nr是人名
rst = analyse.extract_tags(tt, 20, withWeight=False, allowPOS=('nr'))
# allowPos 为默认过滤词性('ns'：地名, 'n：名词', 'vn：名动词', 'v：动词')
rst1 = jieba.analyse.textrank(tt, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
print(type(rst), len(rst))
print(rst)
print(rst1)
#
# 自定义词      参数 词，词频，词性
jieba.add_word('青青的', None, 'a')
jieba.add_word('苍翠的', None, 'a')
s = "五星红旗，随风飘扬。我爱中国。蓝蓝的天空，青青的湖水，苍翠的小山，自由翱翔的雄鹰。"
# 词性标注
words = pseg.cut(s)
print(words)
for word, flag in words:
    print('%s %s' % (word, flag))