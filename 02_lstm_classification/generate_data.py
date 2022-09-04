"""
  * FileName: generate_data.py
  * Author:   Slatter
  * Date:     2022/8/30 15:13
  * Description:
  * History:
"""
import argparse
import os
import json

from utils import write_list_to_file, build_tree

parser = argparse.ArgumentParser(prog="Processing data")
parser.add_argument("--mode", required=True, choices=["b", "f"], help="decide the type of data classification to be "
                                                                      "generated, b for binary, f for fine-grained")
args = parser.parse_args()


def build_vocab(data_path: str, vocab_dir):
    """
    构建词汇表，将词汇表写入文件中
    :param data_path: 训练集路径，其中句子已经做好词语切分
    :param vocab_dir: 所要保存的词汇表路径
    :return: 词汇表构建为token2id的形式，以json格式写入对应路径
    """
    vocab = []
    vocab_path = os.path.join(vocab_dir, "vocab.txt")
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            text = line.strip().lower()  # 去掉换行符, 全部转为小写
            tokens = text.split()
            vocab += tokens
    vocab.append("<unk>")
    vocab.append("<pad>")
    vocab = sorted(list(set(vocab)))  # 去重

    token2id = {}
    for i in range(len(vocab)):
        token2id[vocab[i]] = i

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(token2id, f)
        f.close()
        print("词汇表已生成")


def phrase2label(dict_path: str, check_path: str):
    """
    得到phrase对应的sentiment label
    :param dict_path: phrase映射ids的文件路径
    :param check_path: phrase的ids映射到sentiment label的文件路径，用于判断一句话是属于哪个类别
    :return: phrase到sentiment label的映射
    """
    ids2sentiment = {}
    with open(check_path, 'r', encoding='utf-8') as f:
        # 获取phrase index到sentiment label的映射关系
        lines = f.readlines()[1:]
        for line in lines:
            mapping = line.strip().split("|")
            ids2sentiment[int(mapping[0])] = float(mapping[1])

    phrase2sentiment = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        # 获取phrase到sentiment label的映射关系
        lines = f.readlines()
        for line in lines:
            mapping = line.strip().split("|")  # 这里得到phrase到ids映射
            phrase2sentiment[mapping[0]] = ids2sentiment[int(mapping[1])]

    return phrase2sentiment


def split_data(data_path: str, split_path: str, tree_path: str, dict_path: str, check_path: str, save_dir: str,
               mode: str):
    """
    生成数据
    :param data_path: 数据集路径
    :param split_path: 数据集切分路径
    :param tree_path: 语法树数据所在路径
    :param dict_path: phrase映射ids的文件路径
    :param check_path: phrase的ids映射到sentiment label的文件路径，用于判断一句话是属于哪个类别
    :param save_dir: 所要保存的文件路径
    :param mode: 生成类型 b：二分类  f: 五分类
    :return: 生成train、val、test各数据集的数据
    """
    # train表示原文本， train_tree表示语法树，两个列表一一对应
    train, train_tree, test, test_tree, dev, dev_tree = [], [], [], [], [], []
    idx2label = {}
    with open(split_path, 'r', encoding='utf-8') as f:
        # 获取句子index到数据集label的映射关系
        lines = f.readlines()[1:]
        for line in lines:
            mapping = line.strip().split(",")
            idx2label[int(mapping[0])] = int(mapping[1])

    idx2sentence = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        # 获取句子index到句子的映射关系
        idx = 1  # idx从1开始
        lines = f.readlines()
        for line in lines:
            text = " ".join(line.split("|"))
            idx2sentence[idx] = text
            idx += 1  # 更新idx值

    trees = ["null string"]  # 自动加入一个空行方便按照idx读取，因为idx是从1开始的
    with open(tree_path, 'r', encoding='utf-8') as f:
        # 获取所有句子语法树
        trees += f.readlines()

    if mode == "b":
        # 如果是二分类数据需要拿到phrase到sentiment的映射用于筛掉neutral句子
        phrase2sentiment = phrase2label(dict_path, check_path)

    cnt = 0
    for idx, label in idx2label.items():
        if mode == "b":
            # 如果是二分类，中性句子需要去掉
            if idx2sentence[idx].strip() in phrase2sentiment:
                sentiment = phrase2sentiment[idx2sentence[idx].strip()]  # 为了之后写入方便sentence带了换行符，这里需要去掉一下
                if 0.4 < sentiment <= 0.6:
                    continue
            else:  # 会存在一些phrase并不在dictionary中，这些句子就略过, 看了一下大概有569句的样子
                cnt += 1
                print(idx2sentence[idx].strip())
                continue

        if label == 1:  # train
            train.append(idx2sentence[idx])
            train_tree.append(trees[idx])
        elif label == 2:  # test
            test.append(idx2sentence[idx])
            test_tree.append(trees[idx])
        else:  # dev
            dev.append(idx2sentence[idx])
            dev_tree.append(trees[idx])

    # 目前已经处理好了train, train_tree, test, test_tree, dev, dev_tree，然后将这些数据保存到文件中，train需要进一步处理所以命名有所区别
    train_save_path = os.path.join(save_dir, "trainSentence.txt")
    train_tree_save_path = os.path.join(save_dir, "trainTree.txt")
    dev_save_path = os.path.join(save_dir, "dev.txt")
    dev_tree_save_path = os.path.join(save_dir, "devTree.txt")
    test_save_path = os.path.join(save_dir, "test.txt")
    test_tree_save_path = os.path.join(save_dir, "testTree.txt")

    write_list_to_file(train, train_save_path)
    write_list_to_file(train_tree, train_tree_save_path)
    print("训练集已处理完毕")
    write_list_to_file(dev, dev_save_path)
    write_list_to_file(dev_tree, dev_tree_save_path)
    print("验证集已处理完毕")
    write_list_to_file(test, test_save_path)
    write_list_to_file(test_tree, test_tree_save_path)
    print("测试集已处理完毕")


def generate_phrases(base_dir: str, dict_path: str, check_path: str, mode: str):
    """
    生成最终的短语数据集
    :param base_dir: 数据所在目录
    :param dict_path: phrase映射ids的文件路径
    :param check_path: phrase的ids映射到sentiment label的文件路径，用于判断一句话是属于哪个类别
    :param mode: 生成类型 b：二分类  f: 五分类
    :return:
    """
    assert mode in ["b", "f"], "mode is illegal，please choose from 'b' or 'f', b for binary, f for fine-grained"
    phrase2sentiment = phrase2label(dict_path, check_path)

    # 这里只需要生成train.txt用于训练，dev和test好像是不需要变成短语形式的
    datasets = ["train"]
    for seg in datasets:
        if seg == "train":
            data_path = os.path.join(base_dir, "trainSentence.txt")  # 句子数据集路径
            tree_path = os.path.join(base_dir, "trainTree.txt")  # 语法树数据集路径
            save_path = os.path.join(save_dir, "train.txt")  # 生成的短语数据集保存地址
        elif seg == "dev":
            data_path = os.path.join(base_dir, "devSentence.txt")
            tree_path = os.path.join(base_dir, "devTree.txt")
            save_path = os.path.join(save_dir, "dev.txt")
        else:
            data_path = os.path.join(base_dir, "testSentence.txt")
            tree_path = os.path.join(base_dir, "testTree.txt")
            save_path = os.path.join(save_dir, "test.txt")

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sentences = [line.strip() for line in lines]

        with open(tree_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            trees = [line.strip() for line in lines]

        phrase_dataset = []

        for i in range(len(sentences)):
            sentence, tree = sentences[i], trees[i]
            all_phrases = build_tree(sentence, tree)
            phrase_dataset += all_phrases

        phrase_dataset = list(set(phrase_dataset))  # 再做一次去重
        res = []
        for phrase in phrase_dataset:
            if phrase not in phrase2sentiment:  # 去掉一些没有映射的phrase
                continue
            else:
                if mode == 'b' and 0.4 < phrase2sentiment[phrase] <= 0.6:  # 二分类需要筛掉中性的phrase
                    continue
                res.append(phrase + '\n')  # 加换行符方便写入文件
        write_list_to_file(res, save_path)
        print(f"已生成{seg}.txt")


def generate_label(data_dir: str, dict_path: str, check_path: str, save_dir: str, mode: str):
    """
    给数据集生成label
    :param data_path: 数据集的路径
    :param dict_path: phrase映射ids的文件路径
    :param check_path: phrase的ids映射到sentiment label的文件路径，用于判断一句话是属于哪个类别
    :param save_dir: 生成的label保存地址
    :param mode: 生成类型 b：二分类  f: 五分类
    :return: 将label数据写入对应的文件
    """
    assert mode in ["b", "f"], "mode is illegal，please choose from 'b' or 'f', b for binary, f for fine-grained"

    datasets = ['train', 'dev', 'test']
    if mode == 'b':
        data_dir = os.path.join(data_dir, "binary")
    else:
        data_dir = os.path.join(data_dir, "fine_grained")

    for dataset in datasets:
        data_name = dataset + ".txt"
        label_name = dataset + "label.txt"
        data_path = os.path.join(data_dir, data_name)
        save_path = os.path.join(save_dir, label_name)

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sentences = [line.strip() for line in lines]

        phrase2sentiment = phrase2label(dict_path, check_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                sentiment = phrase2sentiment[sentence]
                if mode == 'b':
                    # 二分类 label 0 表示 negative  1 表示 positive
                    assert 0 <= sentiment <= 0.4 or 0.6 < sentiment <= 1, "二分类数据有误，存在中性phrase"
                    if 0 <= sentiment <= 0.4:
                        label = 0
                    else:
                        label = 1
                else:
                    # 无分类 0,1,2,3,4 分别表示 very negative, negative, neutral, positive, very positive
                    if 0 <= sentiment <= 0.2:
                        label = 0
                    elif 0.2 < sentiment <= 0.4:
                        label = 1
                    elif 0.4 < sentiment <= 0.6:
                        label = 2
                    elif 0.6 < sentiment <= 0.8:
                        label = 3
                    else:
                        label = 4

                f.write(str(label) + '\n')
            print(f"已生成{label_name}")


if __name__ == '__main__':
    data_file = "data/stanfordSentimentTreebank/SOStr.txt"
    split_file = "data/stanfordSentimentTreebank/datasetSplit.txt"
    tree_file = "data/stanfordSentimentTreebank/STree.txt"
    dict_file = "data/stanfordSentimentTreebank/dictionary.txt"
    check_file = "data/stanfordSentimentTreebank/sentiment_labels.txt"
    save_dir = "data"
    mode = args.mode

    if mode == 'b':
        # 二分类数据路径
        save_dir = os.path.join(save_dir, "binary")
    else:
        # 五分类数据路径
        save_dir = os.path.join(save_dir, "fine_grained")

    # 数据预处理，处理过程分为三步：分割数据集、生成训练集词汇表、生成最终的短语数据
    split_data(data_file, split_file, tree_file, dict_file, check_file, save_dir, mode=mode)
    build_vocab(os.path.join(save_dir, "trainSentence.txt"), save_dir)

    # 只需要生成train的phrase集合进行训练即可
    generate_phrases(save_dir, dict_file, check_file, mode=mode)

    # 生成数据集对应的label
    generate_label("data", dict_file, check_file, save_dir, mode=mode)
