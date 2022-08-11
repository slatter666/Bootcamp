"""
预测结果的基类
"""
from typing import List


class TextClassifier(object):

    def classify_text(self, title: List[str], content: List[str]):
        """
        给定新闻的标题和内容，预测新闻的情感极性
        :param title: 新闻的标题
        :param content: 新闻的内容
        :return:
        """
        raise NotImplementedError
