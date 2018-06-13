# -*- coding: utf-8 -*-

import os
import re
import time
import jieba.posseg as pseg


def cut_file_word(read_folder_path, write_folder_path, stop_wordspath):
    """
    news file deal
    :param read_folder_path: 待处理文本的祖父目录
    :param write_folder_path: 处理后的文本保存的祖父目录
    :param stop_wordspath: 停用词表
    :return:
    """

    stopwords = {}.fromkeys([line.rstrip() for line in open(stop_wordspath, "r", encoding='utf-8')])

    # 获取待处理根目录下的所有类别
    folder_list = os.listdir(read_folder_path)

    for folder in folder_list:
        #某类下的路径
        new_folder_path = os.path.join(read_folder_path, folder)

        # 创建保存文件目录
        path = write_folder_path + '/' + folder
        if not os.path.exists(path):
            os.makedirs(path)
            print(path, ' 创建成功')

        save_folder_path = os.path.join(write_folder_path, folder)
        print('---> 请稍等，正在处理中...')

        files = os.listdir(new_folder_path)
        j = 1
        for file in files:
            if j > len(files):
                break
            deal_path = os.path.join(new_folder_path, file)         #处理单个文件的路径
            with open(deal_path, "r", encoding='utf-8') as f:
                txtlist = f.read()

                # 过滤中文、英文标点特殊符号
                # txtlist1 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",txtlist)

            # 带词性标注的分词结果
            words = pseg.cut(txtlist)

            # 单个文本：分词后经停用词处理后的结果
            cut_result = ""
            for word, flag in words:
                if word not in stopwords:
                    cut_result += word + "/" + flag + " "            #去停用词
            savepath = os.path.join(save_folder_path, file)

            get_flag(cut_result, savepath)
            j += 1


def get_flag(cut_result, save_path):
    """
    词性筛选
    :param cut_result: 分词结果
    :param save_path: 保存路径
    :return:
    """

    txtlist=[]
    # 过滤的词性
    tagger = ["/x", "/zg", "/uj", "/ul", "/e", "/d", "/uz", "/y"]

    for line in cut_result.split('\n'):
        # print(line)
        # 将每行数据转为列表
        line_list2 = re.split('[ ]', line)
        # print(line_list2)
        line_list2.append("\n")             # 保持原段落格式存在
        line_list = line_list2[:]
        for segs in line_list2:
            for K in tagger:
                if K in segs:
                    line_list.remove(segs)
                    break

        txtlist.extend(line_list)
    # print(txtlist)
    # 去除词性标签
    flagresult = ""
    for v in txtlist:
        if "/" in v:
            slope = v.index("/")
            letter = v[0:slope] + " "
            flagresult += letter
        else:
            flagresult += v

    # print(flagresult)
    format_data(flagresult, save_path)


def format_data(flag_result, save_path):
    """
    标准化处理，去除空行，空白字符等
    :param flag_result:
    :param save_path:
    :return:
    """
    f2 = open(save_path, "w", encoding='utf-8')
    for line in flag_result.split('\n'):
        if len(line) >= 2:
            line_clean = " ".join(line.split())
            lines = line_clean + " " + "\n"
            f2.write(lines)
    f2.close()


if __name__ == '__main__':

    t1 = time.time()

    # 停用词表
    stop_wordspath = './CNENstopwords.txt'

    # 批量处理文件夹下的文件(300个文本)
    rfolder_path = '/Users/cln/SogouCorpus/FileNews'
    # 分词处理后保存根路径
    wfolder_path = '/Users/cln/SogouCorpus/test'

    # 多文本处理器
    cut_file_word(rfolder_path, wfolder_path, stop_wordspath)

    t2 = time.time()

    print("新闻汉语语料(300个)文本处理完成，耗时：%s 秒" % (t2 - t1))

