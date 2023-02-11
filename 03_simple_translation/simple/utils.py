"""
  * FileName: utils.py
  * Author:   Slatter
  * Date:     2022/12/13 08:26
  * Description:
"""
import random
from typing import List, Dict

from torch.utils.data import Dataset


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
        idx2word[len(word2idx)] = "<sos>"
        word2idx["<sos>"] = len(word2idx)  # start of sentence
        idx2word[len(word2idx)] = "<eos>"
        word2idx["<eos>"] = len(word2idx)  # end of sentence
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


class MTDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        load data from txt file
        :param file_path: data file path
        :return:
        """
        res = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                src, tgt = line.strip().split("|")
                src_tokens = [int(token) for token in src.split()]
                tgt_tokens = [int(token) for token in tgt.split()]
                res.append((src_tokens, tgt_tokens))
            f.close()
        return MTDataset(res)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def check_sentence_pair(src_vocab: Vocab, tgt_vocab: Vocab, dataset, check_idx: int):
    assert len(dataset) != 0, "no data here, make sure you have already load the data"
    assert 0 <= check_idx < len(dataset), "index out of range"

    pair = dataset[check_idx]
    src = " ".join(src_vocab.convert_tokens_to_words(pair[0]))
    tgt = " ".join(tgt_vocab.convert_tokens_to_words(pair[1]))
    print("source sentence:", src)
    print("target sentence:", tgt)


def random_check(src: List[str], tgt: List[str], hyp: List[int], show=20):
    """
    :param src: source sentences list
    :param tgt: target sentences list
    :param hyp: hypothesis sentences list
    :param show: number of results to show
    :return:
    """
    total = len(src)
    for i in range(show):
        idx = random.randint(0, total - 1)
        print('>', src[idx])
        print('=', tgt[idx])
        print('<', hyp[idx])
        print('')


def chop_off_eos(words_list: List[str]):
    """
    chop off the <eos> in the sentence
    :param words_list: words list
    :return:
    """
    if words_list[-1] == '<eos>':
        return words_list[:-1]
    else:
        return words_list


if __name__ == '__main__':
    src_vocab_file = "../data/process_data/de.dict"
    tgt_vocab_file = "../data/process_data/en.dict"
    source_vocab = Vocab.load_from_file(src_vocab_file)
    target_vocab = Vocab.load_from_file(tgt_vocab_file)

    train_file = "../data/process_data/train.txt"
    valid_file = "../data/process_data/dev.txt"
    test_file = "../data/process_data/test.txt"

    train_dataset = MTDataset.load_from_file(train_file)
    valid_dataset = MTDataset.load_from_file(valid_file)
    test_dataset = MTDataset.load_from_file(test_file)

    for i in range(5):
        check_sentence_pair(source_vocab, target_vocab, train_dataset, i)
