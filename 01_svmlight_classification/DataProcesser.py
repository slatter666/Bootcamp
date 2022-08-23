"""
  * FileName: DataProcesser
  * Author:   Slatter
  * Date:     2022/8/17 11:33
  * Description: 
  * History:
"""
import os.path
import numpy as np
from typing import Tuple, List
from sklearn.datasets import fetch_20newsgroups
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer


class DataProcesser():
    def __init__(self, name: str, train_data, train_label, test_data, test_label, feature: str, save_path: str):
        """
        :param name: 数据集名称
        :param train_data: 训练集数据
        :param train_label: 训练集类别标签 二分类为+1 -1 多分类为正整数（不能为0）
        :param test_data: 测试集数据
        :param test_label: 测试集类别标签 二分类为+1 -1 多分类为正整数（不能为0）
        :param feature: 选用特征种类 u: unigram b: bigram ub: unigram+bigram g: glove
        :param save_path: 处理后数据的保存根目录
        """
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.feature = feature
        check_feature = ["u", "b", "ub", "g"]
        assert feature in check_feature, "feature不符合要求，要求值u、b、ub、g，但是输入的feature为:{}".format(feature)

        if feature == "u":
            # unigram
            self.vocab_save_path = "{}/{}/unigram/words.txt".format(save_path, name)
            self.train_save_path = "{}/{}/unigram/train.txt".format(save_path, name)
            self.test_save_path = "{}/{}/unigram/test.txt".format(save_path, name)
            self.vectorizer = TfidfVectorizer(strip_accents="ascii", analyzer="word", stop_words="english",
                                              ngram_range=(1, 1))
        elif feature == "b":
            # bigram
            self.vocab_save_path = "{}/{}/bigram/words.txt".format(save_path, name)
            self.train_save_path = "{}/{}/bigram/train.txt".format(save_path, name)
            self.test_save_path = "{}/{}/bigram/test.txt".format(save_path, name)
            self.vectorizer = TfidfVectorizer(strip_accents="ascii", analyzer="word", stop_words="english",
                                              ngram_range=(2, 2))
        elif feature == "ub":
            # unigram + bigram
            self.vocab_save_path = "{}/{}/ub/words.txt".format(save_path, name)
            self.train_save_path = "{}/{}/ub/train.txt".format(save_path, name)
            self.test_save_path = "{}/{}/ub/test.txt".format(save_path, name)
            self.vectorizer = TfidfVectorizer(strip_accents="ascii", analyzer="word", stop_words="english",
                                              ngram_range=(1, 2))
        else:
            # glove840B 采用glove的时候使用unigram分词
            self.vocab_save_path = "{}/{}/glove/words.txt".format(save_path, name)
            self.train_save_path = "{}/{}/glove/train.txt".format(save_path, name)
            self.test_save_path = "{}/{}/glove/test.txt".format(save_path, name)
            self.vectorizer = TfidfVectorizer(strip_accents="ascii", analyzer="word", stop_words="english",
                                              ngram_range=(1, 1))

    def token2id(self):
        return self.vectorizer.vocabulary_

    def id2token(self):
        token2id = self.token2id()
        res = {}
        for key, value in token2id.items():
            res[value] = key
        return res

    @staticmethod
    def format(coo_matrix):
        row, col, data = coo_matrix.row, coo_matrix.col, coo_matrix.data
        res = []
        for i in range(row.shape[0]):
            res.append((row[i], col[i], data[i]))
        return res

    def get_line(self, label, vector: List[Tuple]) -> str:
        res = str(label)
        if self.feature == "g":
            # glove词向量取平均
            cnt = 0
            id2token = self.id2token()
            sum = np.zeros(300)
            for idx, _ in vector:
                word = id2token[idx]  # 得到对应的词
                if word in self.glove_model:  # 有这个词就加，没有就算了
                    sum += self.glove_model[word]
                    cnt += 1
            if cnt == 0:  # 避免出现除以0的操作
                cnt = 1
            sum /= cnt

            for idx, val in enumerate(sum):
                res += f" {idx + 1}:{val}"  # idx是从0开始的，转换为从1开始
        else:
            vector.sort(key=lambda s: s[0])  # 根据索引值进行从小到大排序（转换为svm_light所需要的格式）
            for idx, val in vector:
                res += f" {idx + 1}:{val}"  # svm_light的feature index是从1开始的而不是0开始，所以索引值加1
        return res

    def process(self):
        train_tfidf = self.vectorizer.fit_transform(self.train_data).tocoo()  # csr_matric转换为coo_matrix
        test_tfidf = self.vectorizer.transform(self.test_data).tocoo()

        # 保存词汇表
        with open(self.vocab_save_path, "w", encoding="utf-8") as f:
            for word in self.vectorizer.vocabulary_.keys():
                f.write(word + '\n')

        if self.feature == "g":
            # glove词向量平均需要加载glove840B
            glove_word2vec = "glove840B/glove.840B.300d.word2vec.txt"
            self.glove_model = KeyedVectors.load_word2vec_format(glove_word2vec, binary=False)

        # 保存训练数据
        with open(self.train_save_path, "w", encoding="utf-8") as f:
            rec = []  # 记录当前一条数据的向量
            values = self.format(train_tfidf)
            before = 0
            for i in range(len(values)):
                row, col, val = values[i]
                if row != before:  # 换行
                    line = self.get_line(label=self.train_label[before], vector=rec)
                    f.write(line + "\n")
                    rec = []
                    before = row
                rec.append((col, val))

            line = self.get_line(label=self.train_label[before], vector=rec)
            f.write(line + "\n")  # 写最后一条数据

        # 保存测试数据
        with open(self.test_save_path, "w", encoding="utf-8") as f:
            rec = []  # 记录当前一条数据的向量
            values = self.format(test_tfidf)
            before = 0
            for i in range(len(values)):
                row, col, val = values[i]
                if row != before:  # 换行
                    line = self.get_line(label=self.test_label[before], vector=rec)
                    f.write(line + "\n")
                    rec = []
                    before = row
                rec.append((col, val))

            line = self.get_line(label=self.test_label[before], vector=rec)
            f.write(line + "\n")  # 写最后一条数据


# if __name__ == '__main__':
#     data_path = "data/raw_data"
#     save_path = "data/processed_data"
#     train_data = fetch_20newsgroups(data_home=data_path, subset="train", remove=("headers", "footers", "quotes")) # 11314
#     test_data = fetch_20newsgroups(data_home=data_path, subset="test", remove=("headers", "footers", "quotes"))  # 7532
#
#     processer = DataProcesser(name="news", train_data=train_data['data'], train_label=train_data['target'],
#                               test_data=test_data['data'], test_label=test_data['target'], feature="u",
#                               save_path=save_path)
