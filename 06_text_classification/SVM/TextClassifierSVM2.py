"""
使用SVM进行预测 去掉停用词  使用tf-idf计算特征权重
1、kernel = 'linear' C=0.98  val_F1 = 0.90  test_F1 = 0.81
2、kernel = 'rbf' C=1 val_F1 = 0.90  test_F1 = 0.82
3、kernel = 'sigmoid' C=0.75 val_F1 = 0.90  test_F1 = 0.81
存在的问题:
1.当前采用的tf进行特征权重的计算, 也许采用if-tdf效果会有所改进
2.对于训练
"""
import re
import os
from typing import List
from TextClassifier import TextClassifier
import json
import jieba
import numpy as np
from sklearn.svm import SVC


class TextClassifierSVM2(TextClassifier):
    def __init__(self, stop_words_path, train_file_path, val_file_path, test_file_path) -> None:
        super().__init__()
        self.stop_words_path = stop_words_path
        self.train_path = train_file_path
        self.val_path = val_file_path
        self.test_path = test_file_path
        self.stop_words = self.load_stop_words()
        self.vocab = self.get_vocab()
        self.word_to_ix = self.get_word_to_ix()

        # svm模型
        self.svm_model = SVC(kernel='rbf', C=1)

    def load_stop_words(self) -> set():
        """
        得到停用词表
        :return: 返回停用词表 类型为set 因为会频繁使用到查找操作
        """
        filelist = os.listdir(self.stop_words_path)
        res = list()
        for dir_path in filelist:
            path = self.stop_words_path + '/' + dir_path
            with open(path, 'r', encoding='UTF-8') as f:
                lines = f.readlines()
                for line in lines:
                    word = re.split('\n', line)[0]
                    res.append(word)
        return set(res)

    def isStop(self, s: str) -> bool:
        """
        返回当前词是否是停用词表中不存在的停用词,主要包含:全字母、数值、百分比等
        :return:
        """
        regex = re.compile(r'[a-zA-Z0-9.%]+')
        res = re.findall(regex, s)
        if res == list():
            return False
        else:
            return True

    def get_vocab(self):
        """
        :return: 返回排序好的、去掉过停用词的词汇表
        """
        res = set()
        with open(self.train_path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
            for mes in data:
                # 文本
                text = mes['text']
                cut_text = jieba.lcut(text)
                for word in cut_text:
                    if not self.isStop(word):  # 不是自定义停用词则加入词汇表中
                        res.add(word)

        # 用<unk>表示不在词语库中的词
        res = res - self.stop_words
        return sorted(list(res))

    def get_word_to_ix(self):
        res = dict()
        for i in range(len(self.vocab)):
            res.setdefault(self.vocab[i], i)
        return res

    def get_transformed_data(self, file=""):
        """
        读取train_data、val_data、test_data
        :return:
        """
        if file == "train":
            path = self.train_path
        elif file == "val":
            path = self.val_path
        elif file == "test":
            path = self.test_path
        else:
            return

        with open(path, 'r', encoding='UTF-8') as f:
            """
                data类型: List
                data中每个元素类型: Dict
            """
            data = json.load(f)
            n = len(data)
            x_matrix = np.zeros((n, len(self.vocab)))  # shape: (n, vocab.size)
            y_matrix = np.zeros(n)  # shape shape (n,)
            idf = np.zeros((1, len(self.vocab)))  # shape (1, vocab.size)   计算idf值

            for i in range(n):
                # 文本
                text = data[i]['text']
                cut_text = jieba.lcut(text)
                for j in range(len(cut_text)):
                    word = cut_text[j]
                    if word in self.stop_words or self.isStop(word) or word not in self.word_to_ix:  # 如果是停用词就不计算
                        continue

                    idx = self.word_to_ix.get(word)  # 得到当前词的下标映射
                    idf[0, idx] += 1
                    x_matrix[i, idx] += 1

                # 文本标签
                label = data[i]['label']
                y_matrix[i] = int(label)

        for i in range(len(self.vocab)):
            if idf[0, i] == 0:
                continue
            else:
                idf[0, i] = np.log(n / idf[0, i])
        log_idf = idf
        x_matrix = x_matrix * log_idf  # 得到最终的tf-idf值

        # 用2-范数求模对数据归一化  即TFC
        p_norm = np.sqrt(np.power(x_matrix, 2).sum(axis=1))  # （x_matrix.shape[0],）
        for i in range(x_matrix.shape[0]):  # 避免出现除以0的情况
            p_norm[i] = 1 if p_norm[i] == 0 else p_norm[i]

        p_norm = p_norm.reshape(x_matrix.shape[0], 1)  # reshape to (x_matrix.shape[0], 1) to fit broadcast
        x_matrix /= p_norm
        return x_matrix, y_matrix

    def train_model(self):
        """
        训练模型
        :return: None
        """
        train_x, train_y = self.get_transformed_data("train")
        self.svm_model.fit(train_x, train_y)

    def predict_val_data(self):
        """
        在验证集上进行预测测试
        :return: None
        """
        val_x, val_y = self.get_transformed_data("val")
        total = val_y.size

        val_pred = self.svm_model.predict(val_x)
        # 计算预测正确的样本数、真正例（TP）、假负例（FN）、假正例（FP）、真负例（TN）
        right = (val_y == val_pred).sum()
        TP, FN, FP, TN = [0] * 4
        for i in range(total):
            if val_y[i] == 1 and val_pred[i] == 1:
                TP += 1
            elif val_y[i] == 1 and val_pred[i] == 0:
                FN += 1
            elif val_y[i] == 0 and val_pred[i] == 1:
                FP += 1
            else:
                TN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

        accuracy = right / total
        print("---------------------------验证集---------------------------")
        print("SVM模型在验证集上准确率为:", accuracy)
        print("SVM模型在验证集上查准率为:", precision)
        print("SVM模型在验证集上查全率为:", recall)
        print("SVM模型在验证集上F1值为:", F1)

    def predict_test_data(self):
        """
        在测试集上进行预测测试
        :return: None
        """
        test_x, test_y = self.get_transformed_data("test")
        total = test_y.size

        test_pred = self.svm_model.predict(test_x)
        # 计算预测正确的样本数、真正例（TP）、假负例（FN）、假正例（FP）、真负例（TN）
        right = (test_y == test_pred).sum()
        TP, FN, FP, TN = [0] * 4
        for i in range(total):
            if test_y[i] == 1 and test_pred[i] == 1:
                TP += 1
            elif test_y[i] == 1 and test_pred[i] == 0:
                FN += 1
            elif test_y[i] == 0 and test_pred[i] == 1:
                FP += 1
            else:
                TN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

        accuracy = right / total
        print("---------------------------测试集---------------------------")
        print("SVM模型在测试集上准确率为:", accuracy)
        print("SVM模型在测试集上查准率为:", precision)
        print("SVM模型在测试集上查全率为:", recall)
        print("SVM模型在测试集上F1值为:", F1)

    def run(self):
        self.train_model()
        self.predict_val_data()
        self.predict_test_data()

    def classify_text(self, title: List[str], content: List[str]) -> List:
        """
        给定新闻的标题和内容，预测新闻的情感极性
        :param title: 新闻的标题
        :param content: 新闻的内容   实际没啥用, 主要是看标题
        :return: 预测结果
        """
        title_matrix = np.zeros((len(title), len(self.vocab)))   # shape: (title.size, vocab.size)
        idf = np.zeros((1, len(self.vocab)))   # shape: (1, vocab.size)
        for i in range(len(title)):
            text = title[i]
            cut_text = jieba.lcut(text)
            for j in range(len(cut_text)):
                word = cut_text[j]
                if word in self.stop_words or self.isStop(word) or word not in self.word_to_ix:
                    continue
                idx = self.word_to_ix.get(word)
                idf[0, idx] += 1
                title_matrix[i, idx] += 1

        for i in range(len(self.vocab)):
            if idf[0, i] == 0:
                continue
            else:
                idf[0, i] = np.log(len(title) / idf[0, i])
        log_idf = idf
        title_matrix = title_matrix * log_idf  # 得到最终的tf-idf值

        p_norm = np.sqrt(np.power(title_matrix, 2).sum(axis=1))  # （title_matrix.shape[0],）
        p_norm = p_norm.reshape(title_matrix.shape[0], 1)  # reshape to (x_matrix.shape[0], 1) to fit broadcast
        title_matrix /= p_norm

        pred = self.svm_model.predict(title_matrix)
        return list(pred)


if __name__ == '__main__':
    stop_words_path = 'stop_words'
    train_data_path = "../../shannon-bootcamp-data/06_text_classification/train_data_v3.json"
    val_data_path = "../../shannon-bootcamp-data/06_text_classification/valid_data_v3.json"
    test_data_path = "../../shannon-bootcamp-data/06_text_classification/test_data_v3.json"
    classifier = TextClassifierSVM2(stop_words_path, train_data_path, val_data_path, test_data_path)
    classifier.run()

    test_titles = [
        '51信用卡CEO孙海涛：“科技+金融”催生金融新世界',
        '美要求对中追加1000亿美元关税(附声明全文)',
        '敲除病虫害基因 让棉花高产又“绿色”',
        '西气东输三线长沙支线工程完工 主要承担向长沙(湘江西)、益阳、常德等地的供气任务',
        '个税变化：元旦后发年终奖 应纳税额有的“打三折”'
    ]
    classify_result = classifier.classify_text(test_titles, [])  # 真实的结果为[0,1,0,0,1]
    print(classify_result)
