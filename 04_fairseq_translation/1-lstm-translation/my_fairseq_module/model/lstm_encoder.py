"""
  * FileName: lstm_encoder.py
  * Author:   Slatter
  * Date:     2022/10/14 21:18
  * Description:  
"""
import torch
from torch import nn
from fairseq import utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from fairseq.models import FairseqEncoder


class LSTMEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_dim=128, hidden_dim=128, layer=1, dropout=0.1):
        super(LSTMEncoder, self).__init__(dictionary)
        self.args = args
        self.layer = layer
        self.embedding = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=True
        )

        self.h_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, src_tokens, src_lengths):
        """

        :param src_tokens: (batch, src_len)
        :param src_lengths: (batch)
        :return:
        """
        # convert left-padding to right-padding
        if self.args.left_pad_source:
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                padding_idx=self.dictionary.pad(),
                left_to_right=True,
            )

        embed = self.embedding(src_tokens)  # (batch, src_len, embed_dim)
        embed = self.dropout(embed)
        embed = pack_padded_sequence(embed, src_lengths.cpu(), batch_first=True)
        # out: (src_len, batch, hidden_dim * 2)   ht, ct: (2 * num_layer, batch, hidden_dim)
        out, (ht, ct) = self.lstm(embed)
        out, _ = pad_packed_sequence(out)
        out = self.o_proj(out.transpose(0, 1))  # (batch, src_len, hidden_dim * 2) -> (batch, src_len, hidden_dim)

        final_hidden, final_cell = [], []
        for i in range(self.layer):
            final_hidden.append(torch.cat((ht[i * 2], ht[i * 2 + 1]), dim=-1))
            final_cell.append(torch.cat((ct[i * 2], ct[i * 2 + 1]), dim=-1))

        final_hidden = self.h_proj(torch.stack(final_hidden))
        final_cell = self.c_proj(torch.stack(final_cell))  # (num_layer, batch, hidden_dim)

        return {
            "final_hidden": final_hidden,
            "final_cell": final_cell,
            "enc_outputs": out,
            "src_len": src_lengths
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        final_hidden = encoder_out["final_hidden"]
        final_cell = encoder_out["final_cell"]
        enc_outputs = encoder_out["enc_outputs"]
        src_len = encoder_out["src_len"]
        return {
            "final_hidden": final_hidden.index_select(1, new_order),
            "final_cell": final_cell.index_select(1, new_order),
            "enc_outputs": enc_outputs.index_select(0, new_order),
            "src_len": src_len.index_select(0, new_order)
        }
