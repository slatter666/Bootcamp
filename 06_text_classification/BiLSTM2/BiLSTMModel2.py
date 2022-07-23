"""
    使用BiLSTM进行分类  用所有LSTM的输出累加然后BatchNormalization之后做为输入进行分类预测
    效果最好的模型参数为embed=400, hidden=140, layer=1
    大概达到 val_acc=0.86, val_F1=0.85, test_F1=0.81

    尝试把单层fc改为MLP之后其实效果也差不多, 该模型的极限大致就是这么多了
    个人认为test_set和train_set、val_set的数据差异还是稍微有点大
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pytorch_lightning as pl


class BiLSTMModel2(pl.LightningModule):
    def __init__(self, vocab_size, embed_size, hidden_size, num_class, num_layers=1, dropout=0.2, lr=0.5, momentum=0.9):
        super(BiLSTMModel2, self).__init__()
        # hyper parameters
        self.LR = lr
        self.Momentum = momentum

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bilstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                              bidirectional=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_size * 2),  # 由于会存在数据分布差异很大的情况, 所以加一层Batch Normalization
            nn.Linear(hidden_size * 2, num_class)
        )

    def forward(self, padded_text, actual_length):
        """
        :param padded_text: 已经转换为tensor的输入文本  (max_len, batch_size)
        :param actual_length: 去掉pad_token后文本的实际长度 (batch)
        :return: logits with shape (b, num_class)
        """
        ### TODO:
        ###     1. Construct Tensor `embed_text` of padded_text with shape (max_len, batch_size, embed_size) using the
        ###        model embeddings
        ###     2. Compute `out`, `ht`, 'ct' by applying the bilstm to `embed_text`.
        ###       - Before you can apply the bilstm, you need to apply the `pack_padded_sequence` function to embed_text.
        ###       - After you apply the bilstm, you need to apply the `pad_packed_sequence` function to out.
        ###       - Note that the shape of the tensor returned by the bilstm is (max_len, b, h * 2)
        ###     3.`out` is a tensor shape (max_len, b, h * 2). We need to compute all the hidden state to get a tensor
        ###        shape (b, h * 2). Then apply the fc layer to this inorder to predict the label

        embed_text = self.embedding(padded_text)
        embed_text = pack_padded_sequence(embed_text, lengths=actual_length, enforce_sorted=False)
        out, (ht, ct) = self.bilstm(embed_text)  # apply bilstm to embed_text
        out, _ = pad_packed_sequence(out)
        result = self.fc(out.sum(dim=0))
        return result

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.LR, momentum=self.Momentum)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # label: (batch_size)  text: (batch_size, max_len) length: (batch_size)
        label, text, length = batch
        # (batch_size, max_len) -> (max_len, batch_size)
        text = text.transpose(0, 1)
        logits = self.forward(text, length)
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
        label, text, length = batch
        text = text.transpose(0, 1)
        logits = self.forward(text, length)
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
        label, text, length = batch
        text = text.transpose(0, 1)
        logits = self.forward(text, length)
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
        label, text, length = batch
        text = text.transpose(0, 1)
        logits = self.forward(text, length)
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
