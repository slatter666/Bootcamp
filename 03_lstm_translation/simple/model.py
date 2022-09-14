"""
# FileName   : model.py
# Author     ：Slatter
# Time       ：2022/9/5 19:36
# Description：
"""
import sys
from typing import List
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
sys.path.append("..")
from utils import Vocab


class SimpleNMT(nn.Module):
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, embed_size: int, hidden_size: int, MAX_LEN: int, device):
        """
        :param src_vocab: source language词汇表大小
        :param tgt_vocab: target language词汇表大小
        :param embed_size: embedding size
        :param hidden_size: hidden size
        :param MAX_LEN: 生成翻译句子的最大长度
        :param device: 设备
        """
        super(SimpleNMT, self).__init__()
        self.MAX_LEN = MAX_LEN
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)

        self.src_embedding = nn.Embedding(self.src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, embed_size)

        # encoder使用BiLSTM, decoder使用LSTMCell逐个生成词
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.h_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.c_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTMCell(embed_size, hidden_size)
        self.o_fc = nn.Linear(hidden_size, len(tgt_vocab))

    def forward(self, source: torch.Tensor, src_length: List[int], target: torch.Tensor, tgt_length: List[int],
                mode: str) -> torch.Tensor:
        """
        :param source: source language tokens list     shape: (batch_size, max_src_len)
        :param src_length: actual source language tokens number    shape: (batch_size)
        :param target: target language tokens list     shape: (batch_size, max_tgt_len)
        :param tgt_length: actual target language tokens number    shape: (batch_size)
        :param mode: train、val、test     we use teacher forcing for training
        :return: dec_logits (batch_size, max_tgt_len or MAX_LEN, tgt_vocab_size)   dec_tokens  (batch_size, MAX_LEN or max_tgt_len)
        """
        assert mode in ["train", "val", "test"], "ERROR: Mode is illegal, choose from train, val, test"
        enc_hidden, enc_cell = self.encode(source, src_length)  # (batch_size, hidden_size)
        dec_logits, dec_tokens = self.decode(target, tgt_length, enc_hidden, enc_cell, mode)
        return dec_logits, dec_tokens

    def encode(self, source: torch.Tensor, lengths: List[int]):
        """
        对source sentence进行编码
        :param source: source sentence tokens list  shape: (batch_size, max_src_len)
        :param lengths: actual sentence length list    shape: (max_src_len)
        :return: final_hidden, final_cell   shape: (batch_size, hidden_size)
        """
        source_embeded = self.src_embedding(source).transpose(0, 1)  # (max_src_len, batch_size, embed_size)
        source_padded = pack_padded_sequence(source_embeded, lengths, enforce_sorted=False)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(source_padded)
        # (2, batch_size, hidden_size) -> (batch_size, hidden_size * 2) -> (batch_size, hidden_size)
        final_hidden = self.h_fc(torch.cat((last_hidden[-2], last_hidden[-1]), dim=-1))
        final_cell = self.c_fc(torch.cat((last_cell[-2], last_hidden[-1]), dim=-1))
        return final_hidden, final_cell

    def decode(self, target: torch.Tensor, lengths: List[int], enc_hidden: torch.Tensor, enc_cell: torch.Tensor,
               mode: str):
        """
        生成翻译句子
        :param target: reference sentence tokens for teacher forcing    shape: (batch_size, max_target_len)
        :param lengths: actual sentence length list         shape: (max_target_len)
        :param enc_hidden: final encoding hidden of encoder   shape: (batch_size, hidden)
        :param enc_cell: final encoding cell of encoder  shape: (batch_size, hidden * 2)
        :param mode: train or test(val)    we use teacher forcing for training
        :return:
        """
        dec_hidden, dec_cell = enc_hidden, enc_cell
        target_embeded = self.tgt_embedding(target).transpose(0, 1)  # (max_tgt_len, batch_size, embed_size)
        res_logits = []
        res_tokens = []

        # 前一个输出 (batch_size, embed_size) 一开始是<SOS>
        o_prev = self.tgt_embedding(
            torch.tensor([self.tgt_vocab.word2idx["<SOS>"]] * enc_hidden.size(0)).to(self.device))
        if mode == "train":
            # 首先生成第一个词然后使用teacher forcing
            dec_hidden, dec_cell = self.decoder(o_prev, (dec_hidden, dec_cell))  # 注意这里传参要用元组
            rec_out = self.o_fc(dec_hidden)
            res_logits.append(rec_out)
            for piece in torch.split(target_embeded, 1):  # (1, batch_size, embed_size)
                o_prev = piece.squeeze(dim=0)  # (batch_size, embed_size)
                dec_hidden, dec_cell = self.decoder(o_prev, (dec_hidden, dec_cell))  # (batch_size, hidden_size)
                rec_out = self.o_fc(dec_hidden)  # (batch_size, tgt_vocab_size)
                res_logits.append(rec_out)
                predict_token = rec_out.argmax(dim=1)
                res_tokens.append(predict_token)
        else:
            for i in range(self.MAX_LEN):
                dec_hidden, dec_cell = self.decoder(o_prev, (dec_hidden, dec_cell))  # (batch_size, hidden_size)
                rec_out = self.o_fc(dec_hidden)  # (batch_size, tgt_vocab_size)
                res_logits.append(rec_out)
                predict_token = rec_out.argmax(dim=1)  # (batch_size)
                res_tokens.append(predict_token)
                o_prev = self.tgt_embedding(predict_token)  # 更新o_prev

        # res_logits和res_tokens类型为List[torch.Tensor], 转为Tensor
        # res_logits  (MAX_LEN or max_tgt_len, batch_size, tgt_vocab_size) -> (batch_size, MAX_LEN or max_tgt_len, tgt_vocab_size)
        res_logits = torch.stack(res_logits, dim=0).transpose(0, 1)  #
        res_tokens = torch.stack(res_tokens, dim=1)  # (batch_size, MAX_LEN or max_tgt_len)
        return res_logits, res_tokens
