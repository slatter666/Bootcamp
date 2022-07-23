"""
  * FileName: ClassificationModel
  * Author:   Slatter
  * Date:     2022/7/22 19:50
  * Description: 使用EmbeddingBag进行模型构建, 不使用LSTM
  * History:
  * <author>          <time>          <version>          <desc>
  * 作者姓名           修改时间           版本号              描述
"""
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, optim


class ClassificationModel(pl.LightningModule):
    def __init__(self, vocab_size, embed_size, num_class, lr=0.5, momentum=0.9):
        super(ClassificationModel, self).__init__()
        # hyper parametes
        self.LR = lr
        self.Momentum = momentum

        self.embedding = nn.EmbeddingBag(vocab_size, embed_size)  # pytorch-lightning好像不支持sparse这样子去做
        self.fc = nn.Linear(embed_size, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.LR, momentum=self.Momentum)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        label, text, offsets = batch
        logits = self.forward(text, offsets)
        train_loss = F.cross_entropy(logits, label)
        train_acc, train_precision, train_recall, train_F1 = self.compute_metrics(logits, label)
        self.log("train_loss", train_loss.item())
        self.log("train_acc", train_acc)
        self.log("train_precision", train_precision)
        self.log("train_recall", train_recall)
        self.log("train_F1", train_F1)
        self.print("train_loss:{:.2f} | train_acc:{:.2f} | train_precison:{:.2f} | train_recall:{:.2f} | train_F1:{"
                   ":.2f}".format(train_loss.item(), train_acc, train_precision, train_recall, train_F1))
        return train_loss

    def validation_step(self, batch, batch_idx):
        label, text, offsets = batch
        logits = self.forward(text, offsets)
        val_loss = F.cross_entropy(logits, label)
        val_acc, val_precision, val_recall, val_F1 = self.compute_metrics(logits, label)
        self.log("val_loss", val_loss.item())
        self.log("val_acc", val_acc)
        self.log("val_precision", val_precision)
        self.log("val_recall", val_recall)
        self.log("val_F1", val_F1)
        self.print("val_loss:{:.2f} | val_acc:{:.2f} | val_precison:{:.2f} | val_recall:{:.2f} | val_F1:{"
                   ":.2f}".format(val_loss.item(), val_acc, val_precision, val_recall, val_F1))

    def test_step(self, batch, batch_idx):
        label, text, offsets = batch
        logits = self.forward(text, offsets)
        test_loss = F.cross_entropy(logits, label)
        test_acc, test_precision, test_recall, test_F1 = self.compute_metrics(logits, label)
        self.log("test_loss", test_loss.item())
        self.log("test_acc", test_acc)
        self.log("test_precision", test_precision)
        self.log("test_recall", test_recall)
        self.log("test_F1", test_F1)
        self.print("test_loss:{:.2f} | test_acc:{:.2f} | test_precison:{:.2f} | test_recall:{:.2f} | test_F1:{"
                   ":.2f}".format(test_loss.item(), test_acc, test_precision, test_recall, test_F1))

    def predict_step(self, batch, batch_idx):
        label, text, offsets = batch
        logits = self.forward(text, offsets)
        predict = F.softmax(logits, dim=1).argmax(dim=1)  # 首先计算预测值
        return predict.tolist()

    def compute_metrics(self, logits, ground_truth):
        """
        :param logits: 经过前向传播得到的logits with shape (b, num_class)
        :param ground_truth: 真实值
        :return: 准确率、查准率、查全率、F1  (acc、precision、recall、F1)
        """
        predict = F.softmax(logits, dim=1).argmax(dim=1)  # 首先计算预测值

        # 计算准确率
        correct = (predict == ground_truth).sum()
        total = predict.size(0)
        acc = (correct / total).item()

        # 计算查全率、查准率、F1
        TP = ((predict == 1) & (ground_truth == 1)).sum().item()
        FP = ((predict == 1) & (ground_truth == 0)).sum().item()
        TN = ((predict == 0) & (ground_truth == 0)).sum().item()
        FN = ((predict == 0) & (ground_truth == 1)).sum().item()
        precision = TP / (TP + FP) if TP != 0 else 0
        recall = TP / (TP + FN) if TP != 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if precision != 0 and recall != 0 else 0
        return acc, precision, recall, F1
