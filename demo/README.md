* stopwords.txt 包含3122个停用词表，包括了中文、英文和特殊符号等
* news_corpus_deal.py 该代码是对搜狗体育新闻预料（300个）的处理，通过jieba分词，词性过滤，去掉停用此，保存以空格分割后的文本
* news_corpus_analy.py 该代码是对处理后的搜狗体育新闻预料，使用nltk进行词频统计，提取出每个文本中出现频率最高的前n个词
* news_corpus_test.py 该代码是对单个文本进行处理，使用nltk进行词频统计，并绘制词频统计图
