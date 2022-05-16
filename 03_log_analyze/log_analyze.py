# encoding utf-8
"""
@author: xiayu
@contact: xiayu_li@shannonai.com
@version: v0.0.1
@file: reg_exp.py
@time: 2019/1/5 12:45
@desc: 测试基础的正则表达式能力
一共三道题目
1、找出log中搜索最多的问题
2、找出log中搜索最多的时间段(精确到小时)
3、找出log中被问到的最多的银行的名称（推荐使用正则）
"""
import json
import re
import time

# log存放的地址

data_path = '../shannon-bootcamp-data/03_log_analyze/osprey_response.log'


class LogAnalyzer(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def most_query(self):
        """
        找出log中搜索最多的问题
        implement here
        """
        query = {}
        with open(self.file_path, encoding='UTF-8') as f_log:
            lines = f_log.readlines()
            for line in lines:
                log = json.loads(line)
                # 问题的提取方式
                log_json = json.loads(log['json'])
                question = log_json['question']
                if query.get(question) is None:
                    query[question] = 1
                else:
                    query[question] += 1
            f_log.close()

        result = max(query, key=query.get)
        return result

    def most_query_time(self):
        """
        找出log中搜索最多的时间段
        implement here
        """
        rec = {}
        with open(self.file_path, encoding='UTF-8') as f_log:
            lines = f_log.readlines()
            for line in lines:
                log = json.loads(line)
                # ts是时间戳 float类型
                ts = log['ts']
                time_format = time.localtime(ts)
                # 精确到小时
                thour = time_format.tm_hour
                if rec.get(thour) is None:
                    rec[thour] = 1
                else:
                    rec[thour] += 1
        max_query_time = max(rec, key=rec.get)
        result = "搜索最多的时间段为:{}点-{}点".format(max_query_time, max_query_time + 1)
        return result

    def most_query_institution(self):
        """
        找出log中被问到的最多的银行的名称
        implement here
        """
        bank_query = {}
        regex = re.compile(r'\w\w银行')
        with open(self.file_path, encoding='UTF-8') as f_log:
            lines = f_log.readlines()
            for line in lines:
                log = json.loads(line)
                # 问题的提取方式
                log_json = json.loads(log['json'])
                question = log_json['question']
                banks = re.findall(regex, question)
                for bank in banks:
                    if bank_query.get(bank) is None:
                        bank_query[bank] = 1
                    else:
                        bank_query[bank] += 1
            f_log.close()

        institution = max(bank_query, key=bank_query.get)
        return institution

    def read_log_sample(self):
        """
        该函数为读取log中的问题和时间的样例
        :return:
        """
        with open(self.file_path, encoding='UTF-8') as f_log:
            lines = f_log.readlines()
            for line in lines:
                log = json.loads(line)
                # 这里的time是时间戳
                time = log['ts']
                print(time)
                # 问题的提取方式
                log_json = json.loads(log['json'])
                question = log_json['question']
                print(question)


def run():
    log_analyzer = LogAnalyzer(data_path)
    print(log_analyzer.most_query())
    print(log_analyzer.most_query_time())
    print(log_analyzer.most_query_institution())


if __name__ == "__main__":
    run()
