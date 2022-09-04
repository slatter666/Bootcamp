"""
# FileName   : run.py
# Author     ：Slatter
# Time       ：2022/9/1 11:18
# Description：
"""
import argparse
import os
import time
from typing import Dict
from functools import partial
from xmlrpc.client import MAXINT

import torch
from torch import nn, optim
from utils import load_vocab, load_dataset, load_embedding
from torch.utils.data import DataLoader
from model import LSTMModel

parser = argparse.ArgumentParser(prog="Run model")
parser.add_argument("--mode", required=True, choices=['b', 'f'],
                    help="decide classification type, b for binary, f for fine-grained")
parser.add_argument("--model-lr", type=float, default=5e-2, help="set model's learning rate")
parser.add_argument("--embed-lr", default=0.1, help="set embedding's learning rate")
parser.add_argument("--decay", type=float, default=1e-4, help="set weight decay")
parser.add_argument("--batch", type=int, default=128, help="set batch size")
parser.add_argument("--epoch", type=int, default=10, help="set training epochs")
parser.add_argument("--embed-size", type=int, default=300, help="set embedding size")
parser.add_argument("--hidden-size", type=int, default=120, help="set hidden size")
parser.add_argument("--layers", type=int, default=1, help="set layers")
parser.add_argument("--dropout", type=float, default=0.9, help="set dropout parameter")
parser.add_argument("--pretrain", default=None, help="if use pretrained word vector, please give the path")
parser.add_argument("--device", required=True, default="cpu", choices=["cpu", "gpu"], help="choose device: gpu or cpu")

args = parser.parse_args()

# 内部设置可用设备进行训练
if args.device == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def collate_batch(batch, vocab: Dict):
    # (phrase, label)
    phrases, length, labels = [], [], []
    for phrase, label in batch:
        labels.append(label)
        words = phrase.lower().split()  # 需要全部转换为小写
        tokens = [vocab.get(word, vocab["<unk>"]) for word in words]
        length.append(len(tokens))
        phrases.append(tokens)

    # padding
    max_len = max(length)
    for i in range(len(phrases)):
        phrases[i] += [vocab["<pad>"]] * (max_len - len(phrases[i]))

    # convert to tensor
    phrases = torch.tensor(phrases, dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)
    return phrases.to(device), length, labels.to(device)


def train(epochs: int, model, optimizer, criterion, train_loader, dev_loader, model_path: str):
    """
    训练
    :param epochs: 训练迭代次数
    :param model: 模型
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param train_loader: 训练集dataloader
    :param dev_loader: 验证集dataloader
    :param model_path: 模型参数保存地址
    :return:
    """
    min_loss = MAXINT
    max_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            phrases, length, labels = batch
            logits = model(phrases, length)
            loss = criterion(logits, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("Training on Epoch:{} | train_loss:{}".format(epoch, train_loss))

        dev_loss, dev_acc = evaluate(model, criterion, dev_loader)
        print("Evaluating on Dev dataset: dev_loss:{:.2f} | accuracy:{:.2f}%".format(dev_loss, dev_acc * 100))

        if dev_loss <= min_loss:
            min_loss = dev_loss
            torch.save(model.state_dict(), model_path)

    print(f"Loss:{min_loss}")
    #     if max_acc <= dev_acc:
    #         max_acc = dev_acc
    #         torch.save(model.state_dict(), 'best.pth')
    #
    # print(f"Accuracy:{max_acc}")


def evaluate(model, criterion, dataloader):
    """
    测试
    :param model: 模型
    :param criterion: 损失函数
    :param dataloader: 测试数据集
    :return: loss、accuracy
    """
    total_loss = 0
    right_count = 0
    total_count = 0
    model.eval()
    for idx, batch in enumerate(dataloader):
        phrases, length, labels = batch
        total_count += len(length)
        logits = model(phrases, length)  # (batch_size, num_class)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        prediction = logits.argmax(dim=1)
        right_count += (prediction == labels).sum().item()

    accuracy = right_count / total_count
    return loss, accuracy


if __name__ == '__main__':
    base_dir = "data"

    mode = args.mode
    model_lr = args.model_lr
    embed_lr = args.embed_lr
    weight_decay = args.decay
    bsz = args.batch
    epochs = args.epoch
    embed_size = args.embed_size
    hidden_size = args.hidden_size
    layers = args.layers
    dropout = args.dropout
    pretrain_path = args.pretrain

    if mode == 'b':
        base_dir = os.path.join(base_dir, "binary")
        num_class = 2
        save_path = "2best.pth"
    else:
        base_dir = os.path.join(base_dir, "fine_grained")
        num_class = 5
        save_path = "5best.pth"

    if pretrain_path is not None:
        vocab, pretrain_weight = load_embedding(pretrain_path, embed_size)
    else:
        # 如果没有预训练的词向量则需要有自己的词汇表
        vocab_file = os.path.join(base_dir, "vocab.txt")
        vocab = load_vocab(vocab_file)

    train_file = os.path.join(base_dir, "train.txt")
    train_label_file = os.path.join(base_dir, "trainlabel.txt")
    dev_file = os.path.join(base_dir, "dev.txt")
    dev_label_file = os.path.join(base_dir, "devlabel.txt")
    test_file = os.path.join(base_dir, "test.txt")
    test_label_file = os.path.join(base_dir, "testlabel.txt")

    train_dataset = load_dataset(train_file, train_label_file)
    dev_dataset = load_dataset(dev_file, dev_label_file)
    test_dataset = load_dataset(test_file, test_label_file)

    collate_fn = partial(collate_batch, vocab=vocab)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bsz, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=bsz, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=bsz, collate_fn=collate_fn)

    if pretrain_path is not None:
        # 有预训练词向量
        model = LSTMModel(vocab_size=len(vocab), embed_size=embed_size, hidden_size=hidden_size, num_class=num_class,
                          layers=layers, dropout=dropout, pretrain=pretrain_weight).to(device)
    else:
        model = LSTMModel(vocab_size=len(vocab), embed_size=embed_size, hidden_size=hidden_size, num_class=num_class,
                          layers=layers, dropout=dropout).to(device)
    optimizer = optim.Adagrad([{'params': model.embedding.parameters(), 'lr': embed_lr},
                               {'params': model.lstm.parameters()},
                               {'params': model.fc.parameters()}], lr=model_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # train
    begin = time.time()
    train(epochs, model, optimizer, criterion, train_dataloader, dev_dataloader, save_path)
    end = time.time()
    print(f"Total training time: {end - begin}s")
    # test
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = evaluate(model, criterion, test_dataloader)
    print("Evaluating on Test dataset: test_loss:{:.2f} | accuracy:{:.2f}%".format(test_loss, test_acc * 100))
