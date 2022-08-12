"""
搜索引擎搭建
"""
import os
import json
from typing import List
import jieba
from collections import defaultdict, Counter
import random


class Indexer(object):
    def __init__(self, doc_path, index_path):
        self.doc_path = doc_path
        self.index_path = index_path
        self.index = defaultdict(set)
        self.process_data()

    def process_data(self):
        # 还专门去测试了一下三种读法的速度  read > readlines > readline
        with open(self.doc_path, 'r', encoding='utf-8') as f:
            lines = f.read().split("\n")
            lines.pop()
            self.build_index(lines)
            f.close()

        # 将数据存入文件中
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f)
            f.close()

    def build_index(self, document_list: List[str]):
        """
        给定document list建立倒排索引
        :param document_list:
        :return:
        """
        for i in range(len(document_list)):
            cut_doc = jieba.lcut(document_list[i])
            for word in cut_doc:
                self.index[word].add(i)

        # 将集合数据转为list
        for key, value in self.index.items():
            self.index[key] = list(value)


class Searcher(object):
    def __init__(self, doc_path, index_path):
        # 加载倒排表以及所有标题数据
        self.index_list = self.load_index(index_path)
        with open(doc_path, 'r', encoding='utf-8') as f:
            self.docs = f.read().split("\n")
            self.docs.pop()

    def load_index(self, index_path: str):
        """
        将index载入内存
        :param index_path:
        :return:
        """
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def filter(self, cut_query: List[str]) -> List[int]:
        """
        多层过滤器，直到最后过滤出来的结果 大于10个就行了  上限就不设置了 好像高不到哪里去  这个数据集不会超过1000的
        :param cut_query: query分词得到的list
        :return: 包含了query中所出现词语的所有文档的list
        """
        potential_doclist = list()
        for word in cut_query:
            potential_doclist += self.index_list.get(word, [])
        docCounter = Counter(potential_doclist)
        maxCnt = docCounter.most_common(1)[0][1]  # res.most_common() = [('a', 3)]  记录最大频次
        rec = defaultdict(list)  # 做一个counter的键值对翻转   类似于正排表转倒排表

        for key, value in docCounter.items():
            rec[value].append(key)

        res = list()
        cnt = 0
        for i in range(maxCnt, 0, -1):
            if cnt >= 10:
                break
            cnt += len(rec[i])
            res.append(rec[i])
        return res

    def search(self, query: str):
        """
        输入一个query，返回10个最相关的结果
        :param query:
        :return:
        """
        cut_query = jieba.lcut(query)
        # 经过filter得到的潜在相关doc_list
        doc_list = self.filter(cut_query)

        level1 = list()  # level1的一定会展示
        level2 = doc_list[-1]  # level2的随机筛出来展示

        # 把之前的结果全放入level1, 因为level1相关性更高且不满10个
        for doc in doc_list[:-1]:
            level1 += doc  # 不用太担心rank的问题，因为这样加一定是按照rank来的

        random_choice = set()
        while len(random_choice) < 10 - len(level1):
            random_choice.add(level2[random.randint(0, len(level2) - 1)])

        final_result_idx = level1 + list(random_choice)
        relevant_docs = list()
        for idx in final_result_idx:
            relevant_docs.append(self.docs[idx])

        return relevant_docs


if __name__ == '__main__':
    news_path = "../shannon-bootcamp-data/05_search_engine/news_title.txt"
    query_path = "../shannon-bootcamp-data/05_search_engine/query.txt"
    index_path = "index.json"

    # 需要写一个函数来判断index文件是否存在
    if not os.path.exists(index_path):
        # 已经存在就不用管了, 如果不存在则生成对应的倒排表
        ins = Indexer(doc_path=news_path, index_path=index_path)  # 耗时大概18s

    with open(query_path, 'r', encoding='utf-8') as f:
        queries = f.read().split("\n")[:-1]

    searcher = Searcher(doc_path=news_path, index_path=index_path)
    while True:
        s = input("\n输入0结束搜索，其他任意字符启动随机搜索：")
        if s == "0":
            break
        else:
            # 随机查找query进行搜索
            query = queries[random.randint(0, len(queries) - 1)]
            result_list = searcher.search(query)
            print("query:", query)
            print("搜索结果如下:", result_list)
