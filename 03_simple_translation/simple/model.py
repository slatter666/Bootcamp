"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2022/12/13 16:27
  * Description:
"""
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, layer=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.layer = layer

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=layer, bidirectional=True, dropout=dropout)
        self.h_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.c_proj = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, source, src_len):
        """
        :param source: (batch, src_len)
        :param src_len: (batch) list
        :return: output, (hidden, cell)
        """
        # (batch, src_len) -> (batch, src_len, embed_size) -> (src_len, batch, embed_size)
        embedded = self.embedding(source).transpose(0, 1)
        embedded = pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        # output: (src_len, batch, hidden_size)  hidden: (layer * 2, batch, hidden_size)
        output, (hidden, cell) = self.lstm(embedded)

        final_hidden, final_cell = [], []
        for i in range(self.layer):
            final_hidden.append(torch.cat((hidden[i * 2], hidden[i * 2 + 1]), dim=-1))
            final_cell.append(torch.cat((cell[i * 2], cell[i * 2 + 1]), dim=-1))

        # (layer, batch, hidden_size * 2) -> (layer, batch, hidden_size)
        final_hidden, final_cell = torch.stack(final_hidden), torch.stack(final_cell)
        final_hidden, final_cell = self.h_proj(final_hidden), self.c_proj(final_cell)
        return output, (final_hidden, final_cell)


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, layer=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=layer, dropout=dropout)
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, prev_out, prev_hidden, prev_cell):
        """
        :param prev_out: (batch, 1)
        :param prev_hidden: (layer, batch, hidden_size)
        :param prev_cell: (layer, batch, hidden_size)
        :return: out, hidden, cell
        """
        # (batch, 1) -> (batch, 1, embed_size) -> (1, batch, embed_size)
        embeded = self.embedding(prev_out).transpose(0, 1)
        # output: (1, batch, hidden_size)  hidden: (layer, batch, hidden_size)
        output, (hidden, cell) = self.lstm(embeded, (prev_hidden, prev_cell))
        out = self.out_proj(output).squeeze(dim=0)  # (1, batch, output_size) -> (batch, output_size)
        return out, hidden, cell


class NMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_size, hidden_size, device, layer=1, dropout=0.1):
        super(NMT, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.teacher_forcing_ratio = 0.5  # 全都使用teacher forcing那么可以让radio=1
        self.device = device
        self.encoder = Encoder(len(src_vocab), embed_size, hidden_size, layer, dropout)
        self.decoder = Decoder(len(tgt_vocab), embed_size, hidden_size, layer)

        # I would like to compute loss in this model
        self.criterion = nn.CrossEntropyLoss()

    def set_teacher_forcing(self, ratio):
        self.teacher_forcing_ratio = ratio

    def forward(self, source, src_len, target, tgt_len):
        """
        :param source: (batch, max_src_len)
        :param src_len: (batch) list
        :param target: (batch, max_tgt_len)
        :param tgt_len: (batch) list
        :return: loss
        """
        batch_size = len(src_len)
        max_tgt_len = target.size(1)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(source, src_len)

        EOS_token = self.tgt_vocab.word2idx['<eos>']
        decoder_input = torch.tensor([[EOS_token]] * batch_size, device=self.device)  # (batch, 1)

        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        outputs = []  # 保存decoder所有的logits输出
        # 决定是否要用teacher forcing
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        for i in range(max_tgt_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs.append(decoder_output)

            # update decoder_input
            if use_teacher_forcing:
                decoder_input = decoder_output.argmax(dim=1).unsqueeze(dim=1)  # (batch) -> (batch, 1)
            else:
                decoder_input = target[:, i].unsqueeze(dim=1)

        outputs = torch.stack(outputs, dim=1)  # (batch, max_tgt_len, output_size)

        # compute loss for batch
        loss = 0
        for i in range(batch_size):
            cur_tgt_len = tgt_len[i]
            loss += self.criterion(outputs[i, :cur_tgt_len], target[i, :cur_tgt_len])

        loss /= batch_size
        return loss

    def generate(self, source, src_len, max_len=30):
        """
        :param source: (batch, max_src_len)  here I would like to set batch=1
        :param src_len: (batch) list
        :param max_len: maximum length to be generate
        :return: tokens
        """
        batch_size = len(src_len)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(source, src_len)

        EOS_token = self.tgt_vocab.word2idx['<eos>']
        decoder_input = torch.tensor([[EOS_token]] * batch_size, device=self.device)  # (batch, 1)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        out_tokens = []  # 保存tokens
        for i in range(max_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)

            token = decoder_output.argmax(dim=1).item()
            out_tokens.append(token)
            if token == EOS_token:
                break

            decoder_input = decoder_output.argmax(dim=1).unsqueeze(dim=1)  # (batch) -> (batch, 1)
        return out_tokens
