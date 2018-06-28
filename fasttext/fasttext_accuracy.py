# -*- coding:utf-8 -*-

import fasttext

# model_path = './data/thucnews.dat.seg.shuf.model.bin'
# test_path = "./data/thucnews.dat.seg.shuf.test"
model_path = './data/sogounews.dat.seg.shuf.model.bin'
test_path = "./data/sogounews.dat.seg.shuf.test"

classifier = fasttext.load_model(model_path, label_prefix='__label__')

result = classifier.test(test_path)
print("准确度：", result.precision)
print("召回率：", result.recall)

labels_right = []
texts = []

with open(test_path, 'r', encoding='utf-8') as testfile:
    for line in testfile:
        line = line.rstrip()
        labels_right.append(line.split("\t")[-1].replace("__label__", ""))

        # predict 的时候，输入的是 list，每一个元素是一个要预测的实例；
        texts.append(line.split("\t")[0])

print(len(texts))
print(classifier.predict(texts), '\n')
# 预测结果为二维形式，输出每一个类别的概率，按概率从大到小排序
labels_predict = [e[0] for e in classifier.predict(texts)]

# print(labels_predict, '\n')

# 去除重复的label
text_labels = list(set(labels_right))
text_predict_labels = list(set(labels_predict))

print("测试的标签类别：", text_labels)
print("预测的标签：", text_predict_labels, '\n')

A = dict.fromkeys(text_labels, 0)           # 预测正确的各个类的数目
B = dict.fromkeys(text_labels, 0)           # 测试集中各个类的数目
C = dict.fromkeys(text_predict_labels, 0)   # 预测结果中各个类的数目

for i in range(0, len(labels_right)):
    B[labels_right[i]] += 1
    C[labels_predict[i]] += 1

    if labels_right[i] == labels_predict[i]:
        A[labels_right[i]] += 1

print("预测正确的类别数：", A)
print("测试集中的类别数：", B)
print("预测结果中的类别数：", C, '\n')

for key in B:
    try:
        # 召回率
        r = float(A[key]) / float(B[key])
        # 精确度
        p = float(A[key]) / float(C[key])
        # F1-Score 度量值
        f = p * r * 2 / (p + r)

        # 类别左对齐，占 15 个字符（为了美观）
        print("%-15s p:%.6f\t r:%f\t f:%f" % (key, p, r, f))
    except:
        print("error:", key, "right:", A.get(key, 0), "real:", B.get(key, 0), "predict:", C.get(key,0))
