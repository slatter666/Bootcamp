# coding: utf-8
"""
@author: gladuo
@contact: me@gladuo.com

@version: 1.0
@file: word_tokenize.py
@time: 2019-03-09 10:30

由于暂时不会基于统计的机器学习算法，因此采用基于词典的分词算法
首先根据dict.txt得到字典
采用正向最大匹配、逆向最大匹配得到结果
最后选取双向最大匹配结果作为分词结果
"""

from typing import List
import os
import json


class Tokenizer(object):
    def __init__(self, dictpath):
        """

        :param dictpath: 词表位置（相对路径）
        """
        self.max_len = -1
        self.word_table_path = dictpath
        self.word_dict_path = "word_dict.txt"
        self.word_dict = {}
        self.get_wordDict()

    def get_wordDict(self):
        if os.path.isfile(self.word_dict_path):
            # 存在词典则已经生成,无需再重复生成
            with open(self.word_dict_path, 'r', encoding='UTF-8') as f:
                self.word_dict = json.load(f)
                f.close()
        else:
            # 还没有生成词典则需进行词典生成
            with open(self.word_table_path, 'r', encoding='UTF-8') as f:
                while True:
                    line = f.readline()
                    if not line or line is None:
                        break
                    word = line.split()[0]
                    self.word_dict.setdefault(len(word), []).append(word)
                f.close()

            with open(self.word_dict_path, 'a+', encoding='UTF-8') as f:
                json.dump(self.word_dict, f)  # 将字典存入json文件中
                f.close()
        keys = [int(i) for i in list(self.word_dict.keys())]
        self.max_len = max(keys)  # 得到最大词长

    def FMM(self, sentence):
        """
        正向最大匹配
        :param sentence: 需要分词的句子
        :return:
        """
        result = []
        index = 0
        length = len(sentence)
        while length > index:
            for i in range(min(length - index, self.max_len), 0, -1):
                word = sentence[index: index + i]
                if word in self.word_dict[str(i)] or i == 1:
                    # 特判1是对词典中不存在的词进行处理
                    index = index + i
                    break
            result.append(word)
        return result

    def BMM(self, sentence):
        """
        逆向最大匹配
        :param sentence: 需要分词的句子
        :return:
        """
        result = []
        index = len(sentence)
        while index > 0:
            if index - self.max_len >= 0:
                begin = index - self.max_len
            else:
                begin = 0

            for i in range(begin, index):
                word = sentence[i:index]
                if word in self.word_dict[str(len(word))] or index - i == 1:
                    index = i
                    break
            result.append(word)
        result.reverse()
        return result

    def MM(self, sentence):
        """
        双向匹配最大匹配
        :param sentence: 需要分词的句子
        :return:
        """
        fmm = self.FMM(sentence)
        bmm = self.BMM(sentence)
        if fmm == bmm:
            return fmm
        elif len(fmm) > len(bmm):
            return bmm
        else:
            return fmm

    def word_tokenize(self, sequence: str) -> List[str]:
        return self.MM(sequence)


def run():
    word_table_path = "../shannon-bootcamp-data/02_word_tokenize/dict.txt"
    tokenizer = Tokenizer(word_table_path)
    print(tokenizer.word_tokenize('欢迎加入香侬科技！'))


if __name__ == '__main__':
    run()
