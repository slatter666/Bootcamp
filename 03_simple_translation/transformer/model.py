"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2022/12/14 12:21
  * Description:  
"""
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embed_size, 2) * math.log(10000) / embed_size)  # (embed_size/2)
        pos = torch.arange(0, maxlen).view(maxlen, 1)  # (maxlen, 1)
        pos_embedding = torch.zeros(maxlen, embed_size)  # (maxlen, embed_size)
        pos_embedding[:, 0::2] = torch.sin(pos * den)  # 偶数位置  (maxlen, embed_size/2)
        pos_embedding[:, 1::2] = torch.cos(pos * den)  # 奇数位置  (maxlen, embed_size/2)
        pos_embedding = pos_embedding.unsqueeze(dim=1)  # (maxlen, 1, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        """
        Add positional embedding to token_embedding
        :param token_embedding: (src_len, batch, embed_size)
        :return:
        """
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])  # 广播到batch


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, tokens: torch.Tensor):
        """
        :param tokens: (src_len, batch)
        :return: embedding (src_len, batch, embed_size)
        """
        return self.embedding(tokens) * math.sqrt(self.embed_size)


class NMT(nn.Module):
    def __init__(self, device, src_vocab, tgt_vocab, embed_size, n_head, enc_layer, dec_layer, dim_feedforward=512,
                 dropout=0.1):
        super(NMT, self).__init__()
        self.device = device
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.embed_size = embed_size

        self.enc_embedding = TokenEmbedding(len(src_vocab), embed_size)
        self.dec_embedding = TokenEmbedding(len(tgt_vocab), embed_size)
        self.pos_embedding = PositionalEncoding(embed_size, dropout)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=n_head,
            num_encoder_layers=enc_layer,
            num_decoder_layers=dec_layer,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc = nn.Linear(embed_size, len(tgt_vocab))

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        """
        :param src: (src_len, batch)
        :param tgt: (tgt_len, batch)
        :param src_mask: (src_len, src_len)
        :param tgt_mask: (tgt_len, tgt_len)
        :param src_padding_mask: (batch, src_len)
        :param tgt_padding_mask: (batch, tgt_len)
        :param memory_key_padding_mask: (batch, src_len)
        :return: logits (tgt_len, batch, tgt_vocab_size)
        """
        src_embeded = self.pos_embedding(self.enc_embedding(src))
        tgt_embeded = self.pos_embedding(self.dec_embedding(tgt))
        outs = self.transformer(src_embeded, tgt_embeded, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.fc(outs)

    def encode(self, src, src_mask):
        src_embeded = self.pos_embedding(self.enc_embedding(src))
        return self.transformer.encoder(src_embeded, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        tgt_embeded = self.pos_embedding(self.dec_embedding(tgt))
        return self.transformer.decoder(tgt_embeded, memory, tgt_mask)

    def greedy_decode(self, src, src_mask, max_len):
        """
        :param src: (src_len, batch)
        :param src_mask: (src_len, src_len)
        :param max_len: max length of sequence to be generated
        :return:
        """
        enc_outputs = self.encode(src, src_mask)  # (src_len, batch, embed_size)

        eos_token = self.tgt_vocab.word2idx['<eos>']
        dec_input = torch.full((1, 1), fill_value=eos_token, device=self.device)  # (tgt_len, batch)

        for i in range(max_len):
            tgt_mask = self.transformer.generate_square_subsequent_mask(dec_input.size(0)).type(torch.bool).to(
                self.device)
            dec_output = self.decode(dec_input, enc_outputs, tgt_mask)  # (tgt_len, batch, embed_size)

            dec_output = dec_output.transpose(0, 1)  # (batch, tgt_len, embed_size)
            logits = self.fc(dec_output[:, -1, :])  # (batch, tgt_vocab_size)
            next_word = torch.argmax(logits, dim=1)  # (batch)

            dec_input = torch.cat([dec_input, torch.full((1, 1), fill_value=next_word[0]).to(self.device)],
                                  dim=0)  # (tgt_len + 1, batch)
            if next_word.item() == self.tgt_vocab.word2idx['<eos>']:
                break

        return dec_input
