"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2022/12/13 20:10
  * Description:  
"""
import random
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        :return: enc_outputs, (hidden, cell)
        """
        # (batch, src_len) -> (batch, src_len, embed_size) -> (src_len, batch, embed_size)
        embedded = self.embedding(source).transpose(0, 1)
        embedded = pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        # output: (src_len, batch, hidden_size)  hidden: (layer * 2, batch, hidden_size)
        output, (hidden, cell) = self.lstm(embedded)
        enc_outputs, _ = pad_packed_sequence(output)
        enc_outputs = enc_outputs.transpose(0, 1)  # (batch, src_len, hidden_size * 2)

        final_hidden, final_cell = [], []
        for i in range(self.layer):
            final_hidden.append(torch.cat((hidden[i * 2], hidden[i * 2 + 1]), dim=-1))
            final_cell.append(torch.cat((cell[i * 2], cell[i * 2 + 1]), dim=-1))

        # (layer, batch, hidden_size * 2) -> (layer, batch, hidden_size)
        final_hidden, final_cell = torch.stack(final_hidden), torch.stack(final_cell)
        final_hidden, final_cell = self.h_proj(final_hidden), self.c_proj(final_cell)
        return enc_outputs, (final_hidden, final_cell)


class Attention(nn.Module):
    def __init__(self, enc_hid_size, dec_hid_size, device, dropout=0.1):
        super(Attention, self).__init__()
        self.device = device
        self.attr = nn.Linear(enc_hid_size, dec_hid_size)
        self.out_proj = nn.Linear(enc_hid_size + dec_hid_size, dec_hid_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, enc_outputs, dec_hidden, src_len):
        """
        :param enc_outputs: encoder outputs (batch, src_len, enc_hid_size)
        :param dec_hidden: (batch, dec_hid_size)
        :param src_len: (batch)  list
        :return: output (batch, dec_hid_size)
        """
        x = self.attr(enc_outputs)  # (batch, src_len, dec_hid_size)
        attr = torch.bmm(x, dec_hidden.unsqueeze(dim=2)).squeeze(dim=2)  # (batch, src_len, 1) -> (batch, src_len)

        enc_masks = self.generate_mask(enc_outputs, src_len)
        # Set e_t to -inf where enc_masks has 1
        attr.data.masked_fill_(enc_masks, -float('inf'))

        # calculate
        prob = torch.softmax(attr, dim=1)  # (batch, src_len)
        prob = prob.unsqueeze(dim=1)  # (batch, 1, src_len)

        t = torch.bmm(prob, enc_outputs).squeeze(dim=1)  # (batch, 1, dec_hid_size) -> (batch, enc_hid_size)
        t = torch.cat((t, dec_hidden), dim=1)  # (batch, enc_hid_size + dec_hid_size)

        out = self.out_proj(t)  # (batch, dec_hid_size)
        out = self.dropout(torch.tanh(out))
        return out

    def generate_mask(self, enc_outputs: torch.Tensor, src_length: List[int]):
        """
        :param enc_outputs: encoder outputs (batch, src_len, enc_hid_size)
        :param src_length: (batch, dec_hid_size)
        :return: enc_masks
        """
        enc_masks = torch.zeros(enc_outputs.size(0), enc_outputs.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_length):
            enc_masks[e_id, src_len:] = 1
        enc_masks = (enc_masks == 1)
        return enc_masks.to(self.device)


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, attention, layer=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=layer, dropout=dropout)
        self.attention = attention
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, prev_out, prev_hidden, prev_cell, enc_outputs, src_len):
        """
        :param prev_out: (batch, 1)
        :param prev_hidden: (layer, batch, hidden_size)
        :param prev_cell: (layer, batch, hidden_size)
        :param enc_outputs: (batch, enc_hid_size) here enc_hid_size = enc_hid_size * 2 because of bidirection
        :param src_len: (batch) list
        :return: out, hidden, cell
        """
        # (batch, 1) -> (batch, 1, embed_size) -> (1, batch, embed_size)
        embeded = self.embedding(prev_out).transpose(0, 1)
        # output: (1, batch, hidden_size)  hidden: (layer, batch, hidden_size)
        output, (hidden, cell) = self.lstm(embeded, (prev_hidden, prev_cell))

        dec_hidden = output.squeeze(dim=0)
        attention = self.attention(enc_outputs, dec_hidden, src_len)  # (batch, dec_hid_size)

        out = self.out_proj(attention)  # (batch, output_size)
        return out, hidden, cell


class NMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_size, hidden_size, device, layer=1, dropout=0.1):
        super(NMT, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.teacher_forcing_ratio = 0.5  # 全都使用teacher forcing那么可以让radio=1
        self.device = device
        self.encoder = Encoder(len(src_vocab), embed_size, hidden_size, layer, dropout)
        self.attention = Attention(hidden_size * 2, hidden_size, device, dropout)
        self.decoder = Decoder(len(tgt_vocab), embed_size, hidden_size, self.attention, layer)

        # I would like to compute loss in this model
        self.criterion = nn.CrossEntropyLoss()

    def set_teacher_forcing(self, ratio):
        self.teacher_forcing_ratio = ratio

    def forward(self, source, src_len, target, tgt_len):
        """
        :param source: source sentences' tokens      (batch, max_src_len)
        :param src_len: actual source sentences' length     (batch) list
        :param target:  target sentences' tokens     (batch, max_tgt_len)
        :param tgt_len: actual target sentences' length    (batch) list
        :return: loss, ppl
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
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs, src_len)
            outputs.append(decoder_output)

            # update decoder_input
            if use_teacher_forcing:
                decoder_input = decoder_output.argmax(dim=1).unsqueeze(dim=1)  # (batch) -> (batch, 1)
            else:
                decoder_input = target[:, i].unsqueeze(dim=1)

        outputs = torch.stack(outputs, dim=1)  # (batch, max_tgt_len, output_size)

        # compute average loss and perplexity for batch
        total_loss = 0
        total_ppl = 0
        for i in range(batch_size):
            cur_tgt_len = tgt_len[i]
            loss = self.criterion(outputs[i, :cur_tgt_len], target[i, :cur_tgt_len])
            total_loss += loss
            total_ppl += torch.exp(loss)   # because here cross entropy has already been divided by cur_tgt_len

        avg_loss = total_loss / batch_size
        avg_ppl = total_ppl / batch_size
        return avg_loss, avg_ppl

    def beam_search_generate(self, source, src_len, beam_size=5, max_len=30):
        """
        Given a single source sentence, perform beam search, yielding translations in the target language
        :param source: (batch, max_src_len)  here I would like to set batch=1
        :param src_len: (batch) list
        :param beam_size: beam size
        :param max_len: maximum length to be generate
        :return: tokens
        """
        batch_size = len(src_len)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(source, src_len)

        EOS_token = self.tgt_vocab.word2idx['<eos>']
        decoder_input = torch.tensor([[EOS_token]] * batch_size, device=self.device)  # (batch, 1)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        prev = [[[], 1, decoder_input, decoder_hidden, decoder_cell]]
        cur_prob = 0
        cur_seq = None  # record sentence that has already finished generating
        for i in range(max_len):
            results = []
            for beam in prev:
                decoder_input, decoder_hidden, decoder_cell = beam[2], beam[3], beam[4]
                # out: (batch, output_size)  hidden, cell: (layer, batch, hidden_size)
                out, hidden, cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs, src_len)
                out = out.squeeze(dim=0)  # (outputsize) cause here batch = 1
                topv, topi = out.topk(beam_size)  # (beam_size)
                for val, idx in zip(topv, topi):
                    gen = beam[0] + [idx.item()]
                    prob = beam[1] * val.item()
                    next_input = idx.unsqueeze(dim=0).unsqueeze(dim=1)  # (batch, 1)
                    results.append([gen, prob, next_input, hidden, cell])

                    if idx.item() == EOS_token and prob > cur_prob:
                        cur_prob = prob
                        cur_seq = gen

            # filter beams
            prev = []
            results.sort(key=lambda x: x[1], reverse=True)
            for res in results:
                if len(prev) == beam_size:
                    break
                prob = res[1]
                if prob > cur_prob:
                    prev.append(res)

            if len(prev) == 0:
                return cur_seq

        if cur_seq is not None:
            return cur_seq
        else:
            prev.sort(key=lambda x: x[1], reverse=True)
            return prev[0][0]
