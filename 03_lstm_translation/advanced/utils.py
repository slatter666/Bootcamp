"""
# FileName   : utils.py
# Author     ：Slatter
# Time       ：2022/9/15 16:15
# Description：
"""
from typing import List, Tuple, Dict
from sacrebleu import corpus_bleu
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
        word2idx["<SOS>"] = len(word2idx)  # start of sentence
        return cls(word2idx, idx2word)

    def get_word_by_token(self, token: int) -> str:
        return self.idx2word[token]

    def get_token_by_word(self, word: str) -> int:
        return self.word2idx[word]

    def convert_tokens_to_words(self, tokens: List[int]) -> List[str]:
        res = []
        for token in tokens:
            res.append(self.get_word_by_token(token))
        return res

    def convert_words_to_tokens(self, words: List[str]) -> List[int]:
        res = []
        for word in words:
            res.append(self.get_token_by_word(word))
        return res


class TranslationDataset(Dataset):
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


def check_sentence(src_vocab: Vocab, tgt_vocab: Vocab, data_path: str, check_idx: int):
    """
    查看翻译句子对
    :param src_vocab: source language vocab
    :param tgt_vocab: target language vocab
    :param data_path: 数据地址
    :param check_num: 查看第idx个翻译句子对
    :return: 输出source sentence和target sentence
    """
    assert check_idx >= 0 and type(check_idx) == int, "ERROR:请输入非负整数!"
    data = TranslationDataset.load_from_file(data_path)
    assert check_idx < len(data), "ERROR:输入的index超出数据范围"
    src_tokens, tgt_tokens = data.__getitem__(check_idx)
    src_words = src_vocab.convert_tokens_to_words(src_tokens)
    tgt_words = tgt_vocab.convert_tokens_to_words(tgt_tokens)

    src_sentence = " ".join(src_words)
    tgt_sentence = " ".join(tgt_words)
    print("source sentence: ", src_sentence)
    print("target sentence: ", tgt_sentence)
    return len(tgt_words)


def format_sentence(vocab: Vocab, target: List[List[int]], length: List[int], generate: List[int]) -> Tuple[List[str], List[str]]:
    """
    将padding后的target tokens去掉<pad>，将generate的句子去掉结尾后的内容
    :param vocab: target language vocabulary
    :param target: reference sentence
    :param length: reference sentence's actual length
    :param generate: generated sentence
    :return: res_tgt: list of references     res_gen: hypothesis
    """
    res_tgt, res_gen = [], []
    for j in range(len(length)):
        # convert padded reference to original reference
        cur_tgt_tokens = target[j][:length[j]]
        cur_tgt_words = vocab.convert_tokens_to_words(cur_tgt_tokens)
        cur_tgt_words.pop()   # 去掉终结符
        res_tgt.append(" ".join(cur_tgt_words))  # 因为只有一条reference所以就这么保存了

        # process generated sentence
        cur_gen_tokens = generate[j]
        cur_gen_words = vocab.convert_tokens_to_words(cur_gen_tokens)
        for k in range(len(cur_gen_words)):
            if cur_gen_words[k] == '.' or cur_gen_words[k] == '?':
                eos = cur_gen_words[k]
                cur_gen_words = cur_gen_words[:k]  # 去掉终结符
                break

        res_gen.append(" ".join(cur_gen_words))
    return res_tgt, res_gen


def compute_bleu(hypo: List[str], ref: List[List[str]]) -> float:
    """
    计算bleu值
    :param hypo: hypothesis sentences
    :param ref: list of references  len(ref) = number of references   len(ref[0]) = total pairs
    :return: bleu score
    """
    return corpus_bleu(hypo, ref)


if __name__ == '__main__':
    src_vocab_file = "../data/process_data/de.dict"
    tgt_vocab_file = "../data/process_data/en.dict"
    source_vocab = Vocab.load_from_file(src_vocab_file)
    target_vocab = Vocab.load_from_file(tgt_vocab_file)

    train_file = "../data/process_data/train.txt"

    res = 0
    nums = 5
    for i in range(nums):
        print(f"-----------------Pair{i}-----------------")
        a = check_sentence(source_vocab, target_vocab, data_path=train_file, check_idx=i)