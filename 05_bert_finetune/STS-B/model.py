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
from scipy.stats import spearmanr
from transformers import BertModel


class STSBert(pl.LightningModule):
    def __init__(self, num_classes, lr, dropout=0.1):
        super(STSBert, self).__init__()
        self.lr = lr
        self.dropout = nn.Dropout(p=dropout)
        self.bert = BertModel.from_pretrained("bert_finetune-base-uncased", mirror='tuna').train()
        self.fc = nn.Linear(768, num_classes)

        self.criterion = nn.MSELoss()

    def forward(self, *args, **kwargs):
        bert_output = self.bert(**kwargs)
        out = self.dropout(self.fc(bert_output.pooler_output))
        out = out.squeeze(dim=1)  # (batch)
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
        prediction = logits.tolist()
        label = label.tolist()
        return prediction, label, val_loss

    def validation_epoch_end(self, outputs):
        pred, truth, loss = [], [], 0
        for step_out in outputs:
            pred += step_out[0]
            truth += step_out[1]
            loss += step_out[2]
        loss /= len(outputs)
        corr, pval = spearmanr(pred, truth)
        self.log('val_loss', loss)
        self.log('val_corr', corr)
        self.print("Val loss: {:.2f} | corr: {:.2f}".format(loss, corr))

    def predict_step(self, batch, batch_idx):
        encoded_input, index = batch
        logits = self.forward(**encoded_input)
        pred = logits.tolist()
        return pred, index.tolist()
