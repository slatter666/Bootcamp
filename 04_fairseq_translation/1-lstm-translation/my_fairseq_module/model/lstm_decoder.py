"""
  * FileName: lstm_decoder.py
  * Author:   Slatter
  * Date:     2022/10/14 21:38
  * Description:  
"""
import torch
from torch import nn
from typing import List
from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.out_proj = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)

    def forward(self, enc_outputs, dec_hidden, src_len):
        """
        :param enc_outputs: encoder outputs (batch, src_len, enc_hid_dim)
        :param dec_hidden: (batch, dec_hid_dim)
        :param src_len: (batch)
        :return: output: (batch, dec_hid_dim)
        """
        # (batch, src_len, 1) -> (batch, src_len)
        attr = torch.bmm(enc_outputs, dec_hidden.unsqueeze(dim=2)).squeeze(dim=2)
        enc_masks = self.generate_mask(enc_outputs, src_len.tolist())

        # set e_t to -inf where enc_masks has 1
        attr.data.masked_fill_(enc_masks, -float('inf'))

        # calculate
        prob = torch.softmax(attr, dim=1)  # (batch, src_len)
        prob = prob.unsqueeze(dim=1)  # (batch, 1, src_len)

        t = torch.bmm(prob, enc_outputs).squeeze(dim=1)  # (batch, enc_hid_dim)
        t = torch.cat((t, dec_hidden), dim=1)  # (batch, enc_hid_dim + dec_hid_dim)

        out = torch.tanh(self.out_proj(t))  # (batch, dec_hid_dim)
        return out

    @staticmethod
    def generate_mask(enc_outputs: torch.Tensor, src_lengths: List[int]):
        """
        :param enc_outputs: encoder outputs (batch, src_len, enc_hid_dim)
        :param src_lengths: (batch, dec_hid_size)
        :return: enc_masks
        """
        enc_masks = torch.zeros(enc_outputs.size(0), enc_outputs.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_lengths):
            enc_masks[e_id, src_len:] = 1
        enc_masks = (enc_masks == 1)
        return enc_masks.to(torch.device('cuda'))


class LSTMDecoder(FairseqIncrementalDecoder):
    def __init__(self, dictionary, encoder_hidden_dim=128, embed_dim=128, hidden_dim=128, layer=1, dropout=0.1):
        super(LSTMDecoder, self).__init__(dictionary)
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=False,
        )
        self.attention = Attention(encoder_hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, len(dictionary))

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        """
        :param prev_output_tokens: previous decoder outputs of shape (batch ,tgt_len), for teacher forcing
        :param encoder_out: output from encoder, used for encoder side attention
        :param incremental_state:
        :return:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        if incremental_state is not None:
            # If the *incremental_state* argument is not ``None`` then we are
            # in incremental inference mode. While *prev_output_tokens* will
            # still contain the entire decoded prefix, we will only use the
            # last step and assume that the rest of the state is cached.
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, tgt_len = prev_output_tokens.size()

        final_hidden = encoder_out['final_hidden']  # (num_layer, batch, encoder_hidden_dim)
        final_cell = encoder_out["final_cell"]
        enc_outputs = encoder_out["enc_outputs"]  # (batch, src_len, hidden_dim)
        src_len = encoder_out["src_len"]  # (batch)

        embed = self.embedding(prev_output_tokens)  # (batch, tgt_len, embed_dim)
        embed = self.dropout(embed)

        # We will now check the cache and load the cached previous hidden and
        # cell states, if they exist, otherwise we will initialize them to
        # zeros (as before). We will use the ``utils.get_incremental_state()``
        # and ``utils.set_incremental_state()`` helpers.
        initial_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )
        if initial_state is None:
            # first time initialization, same as the original version
            initial_state = (
                final_hidden,  # hidden
                final_cell,  # cell
                torch.zeros(bsz, 1, self.hidden_dim).to(torch.device('cuda'))  # prev_out
            )

        dec_output = []
        for i in range(tgt_len):
            cur = embed[:, i, :].unsqueeze(dim=1)  # (batch, 1, embed_dim)
            cur = torch.cat((initial_state[2], cur), dim=2)  # (batch, 1, embed_dim + hidden_dim)
            cur = cur.transpose(0, 1)  # (1, batch, embed_dim + hidden_dim)
            # out: (1, batch, hidden_dim)   ht, ct: (layer, batch, hidden_dim)
            out, (ht, ct) = self.lstm(cur, initial_state[:2])

            out = out.squeeze(dim=0)  # (batch, hidden_dim)
            attention = self.attention(enc_outputs, out, src_len).unsqueeze(dim=1)  # (batch, 1, hidden_dim)
            final_out = self.out_proj(attention)  # (batch, 1, len(dictionary))

            dec_output.append(final_out)
            # update state
            prev_out = attention
            initial_state = (ht, ct, prev_out)

        # Update the cache with the latest hidden and cell states.
        utils.set_incremental_state(
            self, incremental_state, 'prev_state', initial_state,
        )

        output = torch.cat(dec_output, dim=1)
        return output, None

    def reorder_incremental_state(self, incremental_state, new_order):
        # Load the cached state.
        prev_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )

        # Reorder batches according to *new_order*.
        reordered_state = (
            prev_state[0].index_select(1, new_order),  # hidden (num_layer, batch, hidden_dim)
            prev_state[1].index_select(1, new_order),  # cell (num_layer, batch, hidden_dim)
            prev_state[2].index_select(0, new_order)  # prev_out (batch, 1, dec_hid_dim)
        )

        # Update the cached state.
        utils.set_incremental_state(
            self, incremental_state, 'prev_state', reordered_state,
        )
