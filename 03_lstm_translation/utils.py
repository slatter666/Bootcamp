"""
# FileName   : utils.py
# Author     ：Slatter
# Time       ：2022/9/8 11:02
# Description：
"""
from typing import List, Tuple, Dict
from torch.utils.data import Dataset


class Vocab:
    def __init__(self, word2idx: Dict[str, int], idx2word: Dict[int, str]):
        self.word2idx = word2idx
        self.idx2word = idx2word

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def load_from_file(cls, vocab_path: str):
        """从文件中加载词表"""
        word2idx = dict()
        idx2word = dict()
        word2idx["<pad>"] = 0
        idx2word[0] = "<pad>"
        with open(vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                word = lines[i].strip()
                word2idx[word] = i + 1
                idx2word[i + 1] = word

        idx2word[len(word2idx)] = "<SOS>"
        word2idx["<SOS>"] = len(word2idx)   # start of sentence
        return cls(word2idx, idx2word)

    def get_word_by_token(self, token: int) -> str:
        return self.idx2word[token]

    def get_token_by_word(self, word: str) -> int:
        return self.word2idx[word]


class TranslationDataset:
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[List[int], List[int]]:
        return self.pairs[idx]

    @staticmethod
    def cut_sequence(full_seq: str) -> Tuple[List[int], List[int]]:
        """
        将一个seq2seq的tokens分别拆分成两个对应语言的seq
        :param full_seq: seq2seq tokens序列，用|分隔开
        :return: original sequence tokens list, target sequence tokens list
        """
        seqs = full_seq.split("|")
        origin_seq = [int(token) for token in seqs[0].split()]
        target_seq = [int(token) for token in seqs[1].split()]
        return origin_seq, target_seq

    @classmethod
    def load_from_file(cls, file_path: str):
        """从文件导入数据"""
        res = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                src, tgt = cls.cut_sequence(line)
                res.append((src, tgt))
        return cls(res)


def check_sentence(src_vocab: Vocab, tgt_vocab: Vocab, data_path: str, check_num: int):
    """
    查看翻译句子对
    :param src_vocab: source language vocab
    :param tgt_vocab: target language vocab
    :param check_num: 选择查看多少对数据
    :param data_path: 数据地址
    :return: 输出source sentence和target sentence
    """
    assert check_num >= 0 and type(check_num) == int, "WARNING:请输入正整数!"
    data = TranslationDataset.load_from_file(data_path)
    for i in range(check_num):
        print(f"---------------Pair {i}---------------")
        src_tokens, tgt_tokens = data.__getitem__(i)
        src_words, tgt_words = [], []
        for token in src_tokens:
            src_words.append(src_vocab.get_word_by_token(token))

        for token in tgt_tokens:
            tgt_words.append(tgt_vocab.get_word_by_token(token))

        src_sentence = " ".join(src_words)
        tgt_sentence = " ".join(tgt_words)
        print("source sentence: ", src_sentence)
        print("target sentence: ", tgt_sentence)


if __name__ == '__main__':
    src_vocab_file = "data/process_data/de.dict"
    tgt_vocab_file = "data/process_data/en.dict"
    source_vocab = Vocab.load_from_file(src_vocab_file)
    target_vocab = Vocab.load_from_file(tgt_vocab_file)

    train_file = "data/process_data/train.txt"
    check_sentence(source_vocab, target_vocab, data_path=train_file, check_num=5)
