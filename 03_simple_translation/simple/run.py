"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2022/12/13 16:52
  * Description:
"""
import argparse

import torch
from sacrebleu import corpus_bleu
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import NMT
from utils import Vocab, MTDataset, random_check, chop_off_eos

parser = argparse.ArgumentParser(prog="simple seq2seq")
parser.add_argument("--batch", type=int, default=64, help="choose batch size")
parser.add_argument("--embed-size", type=int, default=300, help="choose embedding size")
parser.add_argument("--hidden-size", type=int, default=300, help="choose hidden size")
parser.add_argument("--layer", type=int, default=1, help="choose layer of seq2seq")
parser.add_argument("--dropout", type=float, default=0.1, help="set dropout rate")
parser.add_argument("--max-len", type=int, default=50, help="choose generated translation's max length")
parser.add_argument("--lr", type=float, default=1e-3, help="choose learning rate")
parser.add_argument("--epochs", type=int, default=15, help="choose epoch of training")
parser.add_argument("--mode", type=str, choices=['train', 'test'], help="choose train or test model")

args = parser.parse_args()

# set hyper parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch
epochs = args.epochs
lr = args.lr
embed_size = args.embed_size
hidden_size = args.hidden_size
layer = args.layer
dropout = args.dropout
MAX_LENGTH = args.max_len

# prepare data
src_vocab_file = "../data/process_data/de.dict"
tgt_vocab_file = "../data/process_data/en.dict"
source_vocab = Vocab.load_from_file(src_vocab_file)
target_vocab = Vocab.load_from_file(tgt_vocab_file)

train_file = "../data/process_data/train.txt"
valid_file = "../data/process_data/dev.txt"
test_file = "../data/process_data/test.txt"

train_dataset = MTDataset.load_from_file(train_file)
valid_dataset = MTDataset.load_from_file(valid_file)
test_dataset = MTDataset.load_from_file(test_file)


def collate(batch):
    src, src_len, tgt, tgt_len = [], [], [], []
    for source, target in batch:
        src_tokens = source  # source不加<eos>
        src.append(src_tokens)
        src_len.append(len(src_tokens))
        tgt_tokens = target + [target_vocab.word2idx['<eos>']]
        tgt.append(tgt_tokens)
        tgt_len.append(len(tgt_tokens))

    max_src_len = max(src_len)
    max_tgt_len = max(tgt_len)

    for i in range(len(src)):
        src[i] = src[i] + [source_vocab.word2idx['<pad>']] * (max_src_len - len(src[i]))
        tgt[i] = tgt[i] + [target_vocab.word2idx['<pad>']] * (max_tgt_len - len(tgt[i]))

    src_tensor = torch.tensor(src, dtype=torch.long, device=device)
    tgt_tensor = torch.tensor(tgt, dtype=torch.long, device=device)
    return src_tensor, src_len, tgt_tensor, tgt_len


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate)
val_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate)


def train(model, train_loader, val_loader, learning_rate=0.01):
    min_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, 1 + epochs):
        train_loss = 0
        train_epoch = tqdm(train_loader, total=len(train_loader), leave=True)
        model.train()
        for batch in train_epoch:
            optimizer.zero_grad()
            # src_tensor, tgt_tensor: (batch, max_src_len)  src_len, tgt_len: (batch)
            src_tensor, src_len, tgt_tensor, tgt_len = batch

            loss = model(src_tensor, src_len, tgt_tensor, tgt_len)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_epoch.set_description(f"Epoch [{epoch}/{epochs}]")
            train_epoch.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        val_loss = 0
        val_epoch = tqdm(val_loader, total=len(val_loader), leave=True)
        model.eval()
        with torch.no_grad():
            for batch in val_epoch:
                src_tensor, src_len, tgt_tensor, tgt_len = batch

                loss = model(src_tensor, src_len, tgt_tensor, tgt_len)

                val_loss += loss.item()
                val_epoch.set_description(f"Validating")
                val_epoch.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch}/{epochs}]: Train loss: {train_loss}, Valid loss: {val_loss}")

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')


model = NMT(source_vocab, target_vocab, embed_size, hidden_size, device, layer=layer, dropout=dropout).to(device)

"""
    1.embed_size=300, hidden_size=300 layer=2  MAX_LENGTH=50
        BLEU SCORE: BLEU = 8.59 36.1/11.2/5.5/3.2 (BP = 0.935 ratio = 0.937 hyp_len = 132220 ref_len = 141043)
"""
if args.mode == 'train':
    train(model, train_dataloader, val_dataloader, learning_rate=lr)  # SGD: 0.1 Adam: 0.001

model.load_state_dict(torch.load('best_model.pt'))


def evaluate(model, test_loader):
    src, gen, ref = [], [], []
    model.eval()
    test_epoch = tqdm(test_loader, total=len(test_loader), leave=True)
    with torch.no_grad():
        for batch in test_epoch:
            src_tensor, src_len, tgt_tensor, tgt_len = batch
            tokens = model.generate(src_tensor, src_len, MAX_LENGTH)
            source_sentence = " ".join(chop_off_eos(source_vocab.convert_tokens_to_words(src_tensor[0].tolist())))
            target_sentence = " ".join(chop_off_eos(target_vocab.convert_tokens_to_words(tgt_tensor[0].tolist())))
            output_sentence = " ".join(chop_off_eos(target_vocab.convert_tokens_to_words(tokens)))

            src.append(source_sentence)
            ref.append(target_sentence)
            gen.append(output_sentence)

    bleu = corpus_bleu(gen, [ref])
    print(f"BLEU SCORE: {bleu}")
    return src, ref, gen


test_src, test_ref, test_hyp = evaluate(model, test_dataloader)
random_check(test_src, test_ref, test_hyp)
