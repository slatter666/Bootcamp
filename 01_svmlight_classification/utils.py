"""
  * FileName: utils
  * Author:   Slatter
  * Date:     2022/8/17 16:52
  * Description: 
  * History:
"""
import os


def svm_light_train(svm_path, base_path, train_file, model_file, mode="b", c=1.0, kernel=0):
    """
    训练模型
    :param svm_path: svm_light程序所在地址
    :param base_path: 数据集目录地址
    :param train_file: 训练集地址
    :param model_file: 模型保存地址
    :param mode: 选择任务模式，是二分类任务还是多分类任务
    :param c: 正则化参数
    :param kernel: 采用什么核函数
    :return:
    """
    train_path = base_path + "/" + train_file
    model_path = base_path + "/" + model_file
    if mode == "b":
        # 二分类
        svm_path = svm_path + "/svm_learn"
    else:
        # 多分类
        svm_path = svm_path + "/svm_multiclass_learn"
    order = f"powershell {svm_path} -c {c} -t {kernel} {train_path} {model_path}"  # 因为是在windows下写的，所以调用powershell来运行命令
    os.system(order)


def svm_light_test(svm_path, base_path, test_file, model_file, predict_path, mode="b"):
    """
    测试阶段
    :param svm_path: svm_light程序所在地址
    :param base_path: 数据集目录地址
    :param test_file: 测试集地址
    :param model_file: 模型加载地址
    :param mode: 选择任务模式，是二分类任务还是多分类任务
    :return:
    """
    train_path = base_path + "/" + test_file
    model_path = base_path + "/" + model_file
    predit_path = base_path + "/" + predict_path
    if mode == "b":
        # 二分类
        svm_path = svm_path + "/svm_classify"
    else:
        # 多分类
        svm_path = svm_path + "/svm_multiclass_classify"
    order = f"powershell {svm_path} {train_path} {model_path} {predit_path}"  # 因为是在windows下写的，所以调用powershell来运行命令
    os.system(order)


# 计算多分类的正确率
def compute_acc(base_path, test_file, predict_file):
    """
    :param base_path: 数据目录地址
    :param test_file: 测试文件地址，用于获取ground_truth label
    :param predict_file: 预测文件地址，用于获取predict label
    :return: 正确率accuracy
    """
    test_path = base_path + "/" + test_file
    predict_path = base_path + "/" + predict_file

    with open(test_path, 'r', encoding='utf-8') as f:
        test = f.readlines()

    with open(predict_path, 'r', encoding='utf-8') as f:
        predict = f.readlines()

    test_label = [line.split()[0] for line in test]
    predict_label = [line.split()[0] for line in predict]

    assert len(test_label) == len(predict_label), f"预测的label数和ground_truth label数不一致, 预测:{len(predict_label)} ground_truth:{len(test_label)}"

    total = len(test_label)
    right = 0
    for i in range(total):
        if test_label[i] == predict_label[i]:
            right += 1

    accuracy = right / total
    return accuracy


def compute_F1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


# if __name__ == '__main__':
#     # 测试svm_light_train和svm_light_test
#     svm_path = "svm_light"
#     base_path = "svm_light/example1"
#     train_file = "train.txt"
#     test_file = "test.txt"
#     model_file = "model.txt"
#     predict_path = "predict.txt"
#
#     svm_light_train(svm_path=svm_path, base_path=base_path, train_file=train_file, model_file=model_file)
#     svm_light_test(svm_path=svm_path, base_path=base_path, test_file=test_file, model_file=model_file, predict_path=predict_path)
#
#     # 测试compute_metrics和compute_F1_score
#     base_path = "data/processed_data/news/unigram"
#     test_file = "test.txt"
#     predict_file = "predict.txt"
#     acc = compute_acc(base_path, test_file, predict_file)

