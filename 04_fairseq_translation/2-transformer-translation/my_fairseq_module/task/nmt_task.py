"""
  * FileName: nmt_task.py
  * Author:   Slatter
  * Date:     2023/2/22 00:30
  * Description:  
"""
import os
import json

import numpy
import torch
import numpy as np

from fairseq.data import Dictionary, LanguagePairDataset, FairseqDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig


class NMTDataset(FairseqDataset):
    def __init__(self, tokens_list):
        # 第一版: self.tokens_list = [torch.LongTensor(tokens) for tokens in tokens_list]
        self.tokens_list = [tokens for tokens in tokens_list]
        self.sizes = np.array([len(tokens) for tokens in tokens_list])
        self.size = len(tokens_list)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def __getitem__(self, i):
        self.check_index(i)
        # 第一版: return self.tokens_list[i]
        return torch.LongTensor(self.tokens_list[i])

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]


@register_task('nmt_task')
class NMTTask(TranslationTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('data', metavar='FILE', help='file prefix for data')

    @classmethod
    def setup_task(cls, args, **kwargs):
        cfg = TranslationConfig().from_namespace(args)
        input_vocab = Dictionary.load(os.path.join(cfg.data, 'dict.src.txt'))
        label_vocab = Dictionary.load(os.path.join(cfg.data, 'dict.tgt.txt'))
        print('| [input] dictionary: {} types'.format(len(input_vocab)))
        print('| [label] dictionary: {} types'.format(len(label_vocab)))
        return cls(cfg, input_vocab, label_vocab)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        path = os.path.join(self.cfg.data, f'{split}.json')

        with open(path, 'r') as f:
            data = json.load(f)
        src_tokens_list = []
        tgt_tokens_list = []

        for pair in data:
            src_tokens, tgt_tokens = pair['src'], pair['tgt']
            src_ids = self.src_dict.encode_line(src_tokens, line_tokenizer=lambda x: x, add_if_not_exist=False).tolist()
            tgt_ids = self.tgt_dict.encode_line(tgt_tokens, line_tokenizer=lambda x: x, add_if_not_exist=False).tolist()

            src_tokens_list.append(src_ids)
            tgt_tokens_list.append(tgt_ids)

        src = NMTDataset(src_tokens_list)
        tgt = NMTDataset(tgt_tokens_list)

        self.datasets[split] = LanguagePairDataset(
            src=src,
            src_sizes=src.sizes,
            src_dict=self.src_dict,
            tgt=tgt,
            tgt_sizes=tgt.sizes,
            tgt_dict=self.tgt_dict,
            left_pad_source=False
        )

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict
