"""
  * FileName: generate.py
  * Author:   Slatter
  * Date:     2022/10/13 20:26
  * Description:  
"""
import os
from typing import List, Dict


class Vocab:
    def __init__(self, word2idx: Dict[str, str], idx2word: Dict[str, str]):
        self.word2idx = word2idx
        self.idx2word = idx2word

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def load_from_file(cls, vocab_path: str):
        """从文件中加载词表"""
        word2idx = dict()
        idx2word = dict()
        word2idx['<pad>'] = 0
        idx2word[0] = '<pad>'
        with open(vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, word in enumerate(lines):
                word = word.strip()
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


class TranslationDataset:
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, src_lang, tgt_lang):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.datasets = {"train": [], "valid": [], "test": []}

    @classmethod
    def build_cls(cls, src_vocab, tgt_vocab, src_lang, tgt_lang):
        return TranslationDataset(src_vocab, tgt_vocab, src_lang, tgt_lang)

    def load_dataset(self, split: str, path: str):
        """
        加载数据集
        :param split: train  valid  test
        :param path: 数据集路径
        :return:
        """
        assert split in ["train", "valid", "test"], "split invalid, choose from train, valid, test"
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                src, tgt = line.strip().split("|")
                src_tokens = [int(token) for token in src.split()]
                tgt_tokens = [int(token) for token in tgt.split()]
                src_words = self.src_vocab.convert_tokens_to_words(src_tokens)
                tgt_words = self.tgt_vocab.convert_tokens_to_words(tgt_tokens)
                self.datasets[split].append((src_words, tgt_words))

    def store_format_data(self, split: str, store_dir: str):
        """将数据以特定格式存储起来"""
        assert split in ["train", "valid", "test"], "split invalid, choose from train, valid, test"
        assert len(self.datasets[split]) != 0, "no data here, make sure you have already load the data"
        src_path = os.path.join(store_dir, f"{split}.{self.src_lang}")
        tgt_path = os.path.join(store_dir, f"{split}.{self.tgt_lang}")

        src = open(src_path, 'w', encoding='utf-8')
        tgt = open(tgt_path, 'w', encoding='utf-8')

        for src_words, tgt_words in self.datasets[split]:
            src.write(" ".join(src_words) + "\n")
            tgt.write(" ".join(tgt_words) + "\n")

        src.close()
        tgt.close()

    def check_dataset_size(self):
        return len(self.datasets["train"]), len(self.datasets["valid"]), len(self.datasets["test"])

    def check_sentence_pair(self, split: str, check_idx: int):
        assert split in ["train", "valid", "test"], "split invalid, choose from train, valid, test"
        assert len(self.datasets[split]) != 0, "no data here, make sure you have already load the data"
        assert 0 <= check_idx < len(self.datasets[split]), "index out of range"

        pair = self.datasets[split][check_idx]
        src = " ".join(pair[0])
        tgt = " ".join(pair[1])
        print("source sentence:", src)
        print("target sentence:", tgt)


if __name__ == '__main__':
    src_vocab_file = "../dataset/origin_data/de.dict"
    tgt_vocab_file = "../dataset/origin_data/en.dict"
    source_vocab = Vocab.load_from_file(src_vocab_file)
    target_vocab = Vocab.load_from_file(tgt_vocab_file)

    train_file = "../dataset/origin_data/train.txt"
    valid_file = "../dataset/origin_data/dev.txt"
    test_file = "../dataset/origin_data/test.txt"

    data = TranslationDataset.build_cls(source_vocab, target_vocab, "de", "en")
    data.load_dataset("train", train_file)
    data.load_dataset("valid", valid_file)
    data.load_dataset("test", test_file)

    store_dir = "../dataset/raw_data"
    data.store_format_data("train", store_dir)
    data.store_format_data("valid", store_dir)
    data.store_format_data("test", store_dir)