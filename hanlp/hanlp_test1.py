# -*- coding:utf-8 -*-

import zipfile
import os
from pyhanlp.static import download, remove_file

from pyhanlp import SafeJClass
import time

NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
Fileset = SafeJClass('com.hankcs.hanlp.classification.corpus.FileDataSet')


def test_data_path():
    this_dir = os.getcwd()
    data_path = os.path.join(this_dir, 'data')
    if os.path.isdir(data_path):
        return data_path
    data_path = os.path.join(this_dir[:this_dir.find('pyhanlp')], 'pyhanlp', 'tests', 'data')
    if os.path.isdir(data_path):
        return data_path
    raise FileNotFoundError('找不到测试data目录，请在项目根目录下运行测试脚本')


def ensure_data(data_name, data_url):
    root_path = test_data_path()
    dest_path = os.path.join(root_path, data_name)
    if os.path.exists(dest_path):
        print(dest_path)
        return dest_path
    if data_url.endswith('.zip'):
        dest_path += '.zip'
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with zipfile.ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path[:-len('.zip')]
    return dest_path


def train_or_load_classifier():
    corpus_path = ensure_data('搜狗文本分类语料库迷你版',
                              'http://hanlp.linrunsoft.com/release/corpus/sogou-text-classification-corpus-mini.zip')
    model_path = corpus_path + '.ser'

    fileset = Fileset(corpus_path)
    if os.path.isfile(model_path):
        return NaiveBayesClassifier(IOUtil.readObjectFrom(model_path))
    classifier = NaiveBayesClassifier()
    classifier.train(fileset)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, model_path)


def predict(classifier, text):
    print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))


if __name__ == '__main__':
    t1 = time.time()
    classifier = train_or_load_classifier()
    predict(classifier, "C罗压梅西内马尔蝉联金球奖")
    predict(classifier, "英国造航母耗时8年仍未服役 被中国速度远远甩在身后")
    predict(classifier, "研究生考录模式亟待进一步专业化")
    predict(classifier, "如果真想用食物解压,建议可以食用燕麦")
    predict(classifier, "通用及其部分竞争对手目前正在考虑解决库存问题")
    predict(classifier, "英国简氏防务网站")
    predict(classifier, "一种风险或陷阱")
    # predict(classifier, "孔子是中国古代伟大的思想家、政治家、教育家，儒家学派创始人")
    t2 = time.time()
    print("耗时：", t2 - t1)
