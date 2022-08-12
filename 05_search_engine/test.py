# import numpy as np
# from sklearn import datasets
# from sklearn.svm import SVC
#
# iris = datasets.load_iris()
# X = iris["data"][:, (2, 3)]
# y = iris["target"]
# # 挑选出setosa和versicolor来进行分类
# setosa_or_versicolor = (y == 0) | (y == 1)
# X = X[setosa_or_versicolor]
# y = y[setosa_or_versicolor]
#
# print(X)
# print(y)
#
# # SVM 模型
# svm_model = SVC(kernel="linear", C=float("inf"))
# svm_model.fit(X, y)
# print(svm_model.coef_)
#
# test_x = np.array([[1.4, 0.2]])
# print(test_x)
# print(svm_model.predict(test_x))
#
# def plot_svc_decision_boundary(svm_clf, xmin, xmax):
#     w = svm_model.coef_
#     b = svm_model.intercept_
#     print(w)
#     print(b)

import time
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

news_path = "../shannon-bootcamp-data/05_search_engine/news_title.txt"

begin = time.time()
with open(news_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]


end = time.time()
print("readlines处理总耗时:", end-begin)

begin = time.time()
with open(news_path, 'r', encoding='utf-8') as f:
    lines = f.read().split("\n")
    lines.pop()

end = time.time()
print("readline处理总耗时:", end-begin)