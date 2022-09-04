import json
import os
from typing import List, Dict, Tuple

import torch
from gensim.models import KeyedVectors


def create_file(path: str):
    """
    级联创建文件
    :param path: 文件路径
    :return:
    """
    dir_name = os.path.dirname(path)
    # 如果不存在目录路径则级联创建文件夹
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # 如果不存在文件则创建文件
    if not os.path.exists(path):
        fd = os.open(path, flags=os.O_CREAT)
        os.close(fd=fd)


def write_list_to_file(data: List[str], write_path: str):
    """
    将列表中的句子逐行写入文件
    :param data: list，list中每个元素为一行字符串数据自带换行符
    :param write_path: 要写的路径
    :return:
    """
    if not os.path.exists(write_path):
        create_file(write_path)

    with open(write_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line)
        f.close()


def load_vocab(vocab_path: str) -> Dict:
    """
    加载词汇表
    :param vocab_path: 词汇表路径 token2id形式
    :return: vocab dictionary
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab


def load_embedding(embed_path: str, embed_size):
    """
    导入预训练的词向量
    :param embed_path: 词向量所在地址
    :param embed_size: 词向量维度
    :return: 词汇表、词向量权重
    """
    word2vec = KeyedVectors.load_word2vec_format(embed_path)
    vocab = word2vec.key_to_index
    vocab['<unk>'] = len(word2vec)
    vocab['<pad>'] = len(vocab)

    vocab_size = len(vocab)  # len(word2vec) + 2  增加了<unk>和<pad>
    weight = torch.zeros(vocab_size, embed_size)
    for i in range(len(word2vec)):
        try:
            word = word2vec.index_to_key[i]
            index = vocab[word]
        except:
            continue

        weight[index, :] = torch.from_numpy(word2vec[word].copy())   # 这里必须copy一下，因为word2vec[word]是只读的

    return vocab, weight


def load_dataset(data_path: str, label_path: str) -> List[Tuple]:
    """
    加载数据集用于训练
    :param data_path: 句子数据集路径
    :param label_path: 句子对应的label路径
    :return: list 其中每个元素是一个元组，如("a moving experience", 1)
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = [line.strip() for line in lines]

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        label = [int(line.strip()) for line in lines]

    res = []
    for i in range(len(data)):
        res.append((data[i], label[i]))
    return res


def build_tree(sentence: str, parse_tree: str) -> List[str]:
    """
    根据父指针表示法建数
    :param sentence: 句子，每个token之间用空格隔开 
    :param parse_tree: 语法树，每个树节点之间用|隔开
    :return: 所有phrase列表
    """
    # 因为tree下标是从1开始表示的，所以在第0个位置上随便放个东西占位
    tokens = [""] + sentence.split()
    tree = [0] + parse_tree.split("|")
    tree = [int(node) for node in tree]

    def reconstruct(origin: List[int], visit: List[bool], res: List[List[int]], idx: int):
        """
        用孩子表示法重建树，这里需要递归建树保持phrase的顺序
        :param origin: 原始的树，用父指针表示法表示
        :param visit: 记录哪些结点被遍历过
        :param res就是需要构建的新树
        :param idx: 当前所在结点
        """
        if visit[idx] or idx == 0:
            # 已经遍历过或者遍历到了根结点
            return
        else:
            father = origin[idx]
            res[father].append(idx)
            visit[idx] = True
            reconstruct(origin, visit, res, father)

    visit = [False] * len(tree)
    new_tree = [[] for _ in range(len(tree))]
    for i in range(1, len(tree)):
        reconstruct(tree, visit, new_tree, i)

    nodes = tokens + [""] * (len(tree) - len(tokens))  # nodes就是所有结点表示的phrase
    for i in range(len(tokens), len(new_tree)):
        children = [nodes[child] for child in new_tree[i]]
        nodes[i] = " ".join(children)

    res = list(set(nodes[1:]))  # 去重
    return res


if __name__ == '__main__':
    sentence = "Yet the act is still charming here ."
    parse_tree = "15|13|13|10|9|9|11|12|10|11|12|14|14|15|0"
    phrases = build_tree(sentence, parse_tree)
    print(f"Original Sentence: {sentence}")
    print(f"Parse tree: {parse_tree}")
    print(f"Generated phrases: {phrases}")

    glove = "glove840B/glove.840B.300d.word2vec.txt"
