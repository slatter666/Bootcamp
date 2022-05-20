# encoding utf-8
"""
@author: xiayu
@contact: xiayu_li@shannonai.com
@version: v0.0.1
@file: reg_exp.py
@time: 2019/1/5 12:45
@desc: 测试基础的正则表达式能力
一共五道题目
1、匹配一个人的出生日期（1990年1月2日，1990.1.2， 1990.01.02，1990-01-02）
2、匹配身份证号或者护照号（一个数字或者大写英文，后面结若干数字，最后一个字符可以是英文也可以是数字）
3、匹配电子邮箱中‘@’前面的字符
4、文中出现的百分比（数字加百分号）
5、包含“股东”和“实际控制人”，但是不包含“承诺”
"""
import re

# 测试数据存放的地址
data_path = '../shannon-bootcamp-data/01_reg_exp/reg_exp.txt'


class Regexp(object):
    def __init__(self, file_path=""):
        if file_path == "":
            self.lines = []
        else:
            with open(file_path, encoding='utf-8') as f_e:
                self.lines = f_e.readlines()

    def search_birth_data(self, strs):
        """

        :param strs: 包含生日的string列表
        :return: 生日列表
        """
        result = []
        # 如果写成(?:0?[1-9]|[12]\d|3[01])这种会存在匹配优先级的问题
        regex = re.compile(r'((?:1\d|20)\d{2}[.年-](?:1[0-2]|0?[1-9])[.月-](?:[12]\d|3[01]|0?[1-9])日?)')
        for s in strs:
            temp = re.findall(regex, s)
            result += temp

        return result

    def search_identity_number(self, strs):
        """

        :param strs: 包含证件号码的string列表
        :return: 身份证、护照列表
        """
        result = []
        # regex1用于验证中国身份证  测试使用regex2用于验证身份证和护照 身份证限定18位(17位数+1位校验码) 护照限定9位数(1位字母+8位数)
        regex1 = re.compile(r'((?:[1-6][1-9]|50)\d{4}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx])')
        regex2 = re.compile(r'([1-9][0-9*]{16}[0-9Xx]|[A-Z][0-9*]{8})')
        for s in strs:
            temp = re.findall(regex2, s)
            result += temp

        return result

    def search_email(self, strs):
        """

        :param strs: 包含电子邮箱的string列表
        :return: 电子邮箱前名称列表
        """
        result = []
        regex = re.compile(r'([\w.]+)@')
        for s in strs:
            temp = re.findall(regex, s)
            result += temp

        return result

    def search_percentage(self, strs):
        """

        :param strs: 包含百分比的string列表
        :return: 百分比列表
        """
        result = []
        regex = re.compile(r'([\d.]+%)')
        for s in strs:
            temp = re.findall(regex, s)
            result += temp

        return result

    def search_share_holder(self, strs):
        """

        :param strs: 包含股东的string列表
        :return: 不包含承诺的string列表
        """
        result = []
        regex = re.compile(r'承诺')
        for s in strs:
            temp = re.findall(regex, s)
            if len(temp) == 0:
                result.append(s)

        return result


def run():
    regexp = Regexp(data_path)
    print(regexp.search_birth_data(regexp.lines[:3]))
    print(regexp.search_identity_number(regexp.lines[3:5]))
    print(regexp.search_email(regexp.lines[5:7]))
    print(regexp.search_percentage(regexp.lines[7:8]))
    print(regexp.search_share_holder(regexp.lines[8:]))


if __name__ == '__main__':
    run()