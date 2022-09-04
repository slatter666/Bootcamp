"""
  * FileName   : model.py
  * Author     : Slatter
  * Date       : 2022/9/1 11:05
  * Description:
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_class, layers=1, dropout=0, pretrain=None):
        super(LSTMModel, self).__init__()
        if pretrain == None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings=pretrain)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_class)

    def forward(self, x, length):
        """
        :param x: phrase (batch, max_len)
        :param length: phrase length corresponding to phrase
        :return: logits (batch, num_class)
        """
        x = x.transpose(0, 1)  # (batch, max_len) -> (max_len, batch)
        embed = self.embedding(x)  # (max_len, batch, embed_size)
        embed = pack_padded_sequence(input=embed, lengths=length, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(embed)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # (batch, hidden_size * 2)
        out = self.fc(final_hidden)
        return out
