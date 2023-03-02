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


class MRPCBert(pl.LightningModule):
    def __init__(self, num_classes, lr, dropout=0.1):
        super(MRPCBert, self).__init__()
        self.lr = lr
        self.dropout = nn.Dropout(p=dropout)
        self.bert = BertModel.from_pretrained("bert_finetune-base-uncased", mirror='tuna').train()
        self.fc = nn.Linear(768, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        bert_output = self.bert(**kwargs)
        out = self.dropout(self.fc(bert_output.pooler_output))
        return out

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        encoded_input, label = batch
        logits = self.forward(**encoded_input)
        train_loss = self.criterion(logits, label)
        return train_loss

    def validation_step(self, batch, batch_idx):
        encoded_input, label = batch
        logits = self.forward(**encoded_input)
        val_loss = self.criterion(logits, label).item()
        prediction = self.get_prediction(logits).tolist()
        label = label.tolist()
        return prediction, label, val_loss

    def validation_epoch_end(self, outputs):
        pred, truth, loss = [], [], 0
        for step_out in outputs:
            pred += step_out[0]
            truth += step_out[1]
            loss += step_out[2]
        loss /= len(outputs)
        acc, pre, rec, f1 = accuracy_score(truth, pred), precision_score(truth, pred), recall_score(truth, pred), f1_score(truth, pred)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_pre', pre)
        self.log('val_rec', rec)
        self.log('val_f1', f1)
        self.print("Val loss: {:.2f} | acc: {:.2f} | pre: {:.2f} | rec: {:.2f} | f1: {:.2f}".format(loss, acc, pre, rec, f1))

    def predict_step(self, batch, batch_idx):
        encoded_input, index = batch
        logits = self.forward(**encoded_input)
        pred = self.get_prediction(logits).tolist()
        return pred, index.tolist()

    @staticmethod
    def get_prediction(logits: torch.Tensor):
        """
        :param logits: (batch, num_classed)
        :return:
        """
        return logits.argmax(dim=1)
