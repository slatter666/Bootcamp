"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/2/25 14:26
  * Description:  
"""
from typing import Any

import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, optim
from transformers import BertModel


class NERBert(pl.LightningModule):
    def __init__(self, num_classes, lr, dropout=0.1):
        super(NERBert, self).__init__()
        self.lr = lr
        self.bert = BertModel.from_pretrained("bert_finetune-base-cased", mirror='tuna')
        # for param in self.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(768, num_classes)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, *args, **kwargs):
        bert_output = self.bert(**kwargs)
        hidden = bert_output.last_hidden_state[:, 1:-1, :]  # chop off the [CLS] and [SEP]
        out = self.dropout(self.fc(hidden))
        return out

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        encoded_input, label = batch
        logits = self.forward(**encoded_input)
        train_loss = self.criterion(logits.transpose(1, 2), label)
        return train_loss

    def validation_step(self, batch, batch_idx):
        encoded_input, label = batch
        logits = self.forward(**encoded_input)
        val_loss = self.criterion(logits.transpose(1, 2), label).item()
        prediction, label = self.extract_result(logits, label)
        return prediction, label, val_loss

    def validation_epoch_end(self, outputs):
        pred, truth, loss = [], [], 0
        for step_out in outputs:
            pred += step_out[0]
            truth += step_out[1]
            loss += step_out[2]
        loss /= len(outputs)
        acc, f1 = accuracy_score(truth, pred), f1_score(truth, pred, average='micro')
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_f1', f1)
        self.print("Val loss: {:.2f} | acc: {:.2f} | f1: {:.2f}".format(loss, acc, f1))

    def test_step(self, batch, batch_idx):
        encoded_input, label = batch
        logits = self.forward(**encoded_input)
        val_loss = self.criterion(logits.transpose(1, 2), label).item()
        prediction, label = self.extract_result(logits, label)
        return prediction, label, val_loss

    def test_epoch_end(self, outputs):
        pred, truth, loss = [], [], 0
        for step_out in outputs:
            pred += step_out[0]
            truth += step_out[1]
            loss += step_out[2]
        loss /= len(outputs)
        acc, f1 = accuracy_score(truth, pred), f1_score(truth, pred, average='micro')
        self.print("Test loss: {:.2f} | acc: {:.2f} | f1: {:.3f}".format(loss, acc, f1))

    def predict(self, enc_inputs):
        logits = self.forward(**enc_inputs)
        predition = logits.argmax(dim=-1).tolist()[0]
        return predition

    @staticmethod
    def extract_result(logits: torch.Tensor, label: torch.Tensor):
        """
        :param logits: (batch, num_classed)
        :return:
        """
        pred = logits.argmax(dim=-1).tolist()
        label = label.tolist()

        predition, truth = [], []
        for i in range(len(pred)):
            predition += pred[i]
            truth += label[i]

        res_pred, res_label = [], []
        for i in range(len(truth)):
            if truth[i] == -1:
                continue
            else:
                res_pred.append(predition[i])
                res_label.append(truth[i])

        return res_pred, res_label
