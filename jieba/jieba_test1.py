# -*- coding: utf-8 -*-

import jieba
import os

# 多进程并行处理
jieba.enable_parallel(4)
text = open("../testdata/female.txt", 'rb')
des_dir = "../testdata/test"

tt = text.read()
jj = jieba.cut(tt)
cc = "/ ".join(jj)

exit_code = os.system("echo '%s' > %s" % (cc, des_dir))

print(exit_code)
print(type(jj), type(str(tt)))

# generator 转 list
print(list(jieba.cut("这仅仅是一个测试。")))
print(list(jieba.cut(str(tt, encoding="utf-8"))))
