"""
# FileName   : run.py
# Author     ：Slatter
# Time       ：2022/9/12 12:15
# Description：
"""
import sys

import nltk.translate.bleu_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from functools import partial
from model import SimpleNMT
from xmlrpc.client import MAXINT
import argparse

sys.path.append("..")
from utils import Vocab, TranslationDataset, format_sentence, compute_bleu

parser = argparse.ArgumentParser(prog="simple seq2seq")
parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="choose device to train the model, cpu or gpu")
parser.add_argument("--batch", type=int, default=32, help="choose batch size")
parser.add_argument("--embed-size", type=int, default=200, help="choose embedding size")
parser.add_argument("--hidden-size", type=int, default=120, help="choose hidden size")
parser.add_argument("--max-len", type=int, default=50, help="choose generated translation's max length")
parser.add_argument("--lr", type=float, default=0.05, help="choose learning rate")
parser.add_argument("--weight-decay", type=float, default=0.01, help="choose weight decay")
parser.add_argument("--epoch", type=int, default=5, help="choose epoch of training")

args = parser.parse_args()

if args.device == "gpu":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")
BATCH_SIZE = args.batch
EMBED_SIZE = args.embed_size
HIDDEN_SIZE = args.hidden_size
MAX_LEN = args.max_len
LR = args.lr
WEIGHT_DECAY = args.weight_decay
EPOCH = args.epoch

src_vocab_file = "../data/process_data/de.dict"
tgt_vocab_file = "../data/process_data/en.dict"
source_vocab = Vocab.load_from_file(src_vocab_file)
target_vocab = Vocab.load_from_file(tgt_vocab_file)


def collate_batch(batch, src_vocab, tgt_vocab):
    max_src_len, max_tgt_len = -1, -1
    src_batch, tgt_batch = [], []
    src_lengths, tgt_lengths = [], []
    for source, target in batch:
        max_src_len = max(max_src_len, len(source))
        max_tgt_len = max(max_tgt_len, len(target))
        src_batch.append(source)
        tgt_batch.append(target)
        src_lengths.append(len(source))
        tgt_lengths.append(len(target))

    # padding
    for i in range(len(src_batch)):
        src_batch[i] = src_batch[i] + [src_vocab.word2idx["<pad>"]] * (max_src_len - len(src_batch[i]))
        tgt_batch[i] = tgt_batch[i] + [tgt_vocab.word2idx["<pad>"]] * (max_tgt_len - len(tgt_batch[i]))

    src_batch = torch.tensor(src_batch, dtype=torch.int64).to(DEVICE)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64).to(DEVICE)
    return src_batch, src_lengths, tgt_batch, tgt_lengths


train_file = "../data/process_data/train.txt"
dev_file = "../data/process_data/dev.txt"
test_file = "../data/process_data/test.txt"
train_dataset = TranslationDataset.load_from_file(train_file)
dev_dataset = TranslationDataset.load_from_file(dev_file)
test_dataset = TranslationDataset.load_from_file(test_file)

collate_fn = partial(collate_batch, src_vocab=source_vocab, tgt_vocab=target_vocab)  # 这里必须一对一指定参数
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = SimpleNMT(source_vocab, target_vocab, EMBED_SIZE, HIDDEN_SIZE, MAX_LEN, DEVICE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()


def compute_batch_loss(logits, target, lengths):
    """
    计算loss
    :param logits: 计算得到的logits值    (batch_size, MAX_LEN or max_tgt_Len, tgt_vocab_size)
    :param target: 实际token值       (batch_size, max_tgt_len)
    :param lengths: 实际tokens长度   (batch_size)
    :return:
    """
    total_loss = torch.tensor(0, dtype=torch.float32).to(DEVICE)
    for i in range(len(lengths)):
        translation = logits[i, :lengths[i], :]  # (actual_tgt_len, tgt_vocab_size)
        ground_truth = target[i, :lengths[i]]  # (actual_tgt_len)
        loss = criterion(translation, ground_truth)
        total_loss += loss
    return total_loss


def train():
    min_loss = MAXINT
    for i in range(1, EPOCH + 1):
        train_loss = 0
        model.train()
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            source, src_lengths, target, tgt_lengths = batch
            logits, tokens = model(source, src_lengths, target, tgt_lengths, mode="train")
            loss = compute_batch_loss(logits, target, tgt_lengths)
            train_loss += loss

            loss.backward()
            optimizer.step()
            print("Train Epoch{}, iter: {}/{}: iter_loss:{:.2f}".format(i, idx, len(train_dataloader), loss))

        print("Train Epoch{}: train_loss:{:.2f}".format(i, train_loss))

        with torch.no_grad():
            val_loss = 0
            model.eval()
            for idx, batch in enumerate(val_dataloader):
                source, src_lengths, target, tgt_lengths = batch
                logits, tokens = model(source, src_lengths, target, tgt_lengths, mode="val")
                loss = compute_batch_loss(logits, target, tgt_lengths)
                val_loss += loss

        print("Epoch{}: val_loss:{:.2f}".format(i, val_loss))

        if min_loss > val_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), 'best.pth')

    print("The best model's loss:", min_loss)


train()   # 训练完成后请注释此行代码

model.load_state_dict(torch.load("best.pth"))


def test():
    rec_ref, rec_gen = [], []
    with torch.no_grad():
        test_loss = 0
        model.eval()
        for idx, batch in enumerate(test_dataloader):
            source, src_lengths, target, tgt_lengths = batch
            logits, tokens = model(source, src_lengths, target, tgt_lengths, mode="test")
            loss = compute_batch_loss(logits, target, tgt_lengths)
            test_loss += loss

            # 转换数据用于计算bleu-score
            t1, t2 = format_sentence(target_vocab, target.tolist(), tgt_lengths, tokens.tolist())
            rec_ref += t1
            rec_gen += t2
        # 测试集计算平均loss
        print("average test loss:{:.2f}".format(test_loss / len(test_dataloader)))

    # 计算bleu值
    bleu_score = compute_bleu(rec_ref, rec_gen)
    print("BLEU SCORE: {}".format(bleu_score))


test()
