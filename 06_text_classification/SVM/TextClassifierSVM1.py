"""
使用SVM进行预测 不去掉停用词 使用tf计算特征权重
1、kernel = 'linear' C=10  val_F1 = 0.88  test_F1 = 0.78
2、kernel = 'rbf' C=1 val_F1 = 0.89  test_F1 = 0.78
3、kernel = 'sigmoid' C=1 val_F1 = 0.88  test_F1 = 0.80
"""
from typing import List
from TextClassifier import TextClassifier
import json
import jieba
import numpy as np
from sklearn.svm import SVC


class TextClassifierSVM1(TextClassifier):
    def __init__(self, train_file_path, val_file_path, test_file_path) -> None:
        super().__init__()
        self.train_path = train_file_path
        self.val_path = val_file_path
        self.test_path = test_file_path
        self.vocab = self.get_vocab()
        self.word_to_ix = self.get_word_to_ix()

        # svm模型
        self.svm_model = SVC(kernel='sigmoid', C=1)

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
                temp = set(jieba.lcut(text))
                res = set.union(res, temp)
        # 用<unk>表示不在词语库中的词
        res.add('<unk>')
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
            x_matrix = np.zeros((n, len(self.vocab)))
            y_matrix = np.zeros(n)
            for i in range(n):
                # 文本
                text = data[i]['text']
                cut_text = jieba.lcut(text)
                for j in range(len(cut_text)):
                    word = cut_text[j]
                    idx = self.word_to_ix.get(word, self.word_to_ix['<unk>'])  # 得到当前词的下标映射
                    x_matrix[i, idx] += 1

                # 文本标签
                label = data[i]['label']
                y_matrix[i] = int(label)

        # 用2-范数求模对数据归一化
        p_norm = np.sqrt(np.power(x_matrix, 2).sum(axis=1))  # （x_matrix.shape[0],）
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
        title_matrix = np.zeros((len(title), len(self.vocab)))
        for i in range(len(title)):
            text = title[i]
            cut_text = jieba.lcut(text)
            for j in range(len(cut_text)):
                word = cut_text[j]
                idx = self.word_to_ix.get(word, self.word_to_ix['<unk>'])
                title_matrix[i, idx] += 1

        p_norm = np.sqrt(np.power(title_matrix, 2).sum(axis=1))  # （x_matrix.shape[0],）
        p_norm = p_norm.reshape(title_matrix.shape[0], 1)  # reshape to (x_matrix.shape[0], 1) to fit broadcast
        title_matrix /= p_norm

        pred = self.svm_model.predict(title_matrix)
        return list(pred)


if __name__ == '__main__':
    train_data_path = "../../shannon-bootcamp-data/06_text_classification/train_data_v3.json"
    val_data_path = "../../shannon-bootcamp-data/06_text_classification/valid_data_v3.json"
    test_data_path = "../../shannon-bootcamp-data/06_text_classification/test_data_v3.json"
    classifier = TextClassifierSVM1(train_data_path, val_data_path, test_data_path)
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
