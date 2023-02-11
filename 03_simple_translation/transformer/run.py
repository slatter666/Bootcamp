"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2022/12/14 18:16
  * Description:
"""
import argparse

import torch
from sacrebleu import corpus_bleu
from torch import optim, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import NMT
from utils import Vocab, MTDataset, random_check, chop_off_eos

parser = argparse.ArgumentParser(prog="advance seq2seq")
parser.add_argument("--batch", type=int, default=128, help="choose batch size")
parser.add_argument("--embed-size", type=int, default=512, help="embedding size")
parser.add_argument("--ffn-hid-size", type=int, default=512, help="feed-forward hidden size")
parser.add_argument("--nhead", type=int, default=8, help="num of head")
parser.add_argument("--encoder-layer", type=int, default=3, help="num of seq2seq encoder layer")
parser.add_argument("--decoder-layer", type=int, default=3, help="num of seq2seq decoder layer")
parser.add_argument("--dropout", type=float, default=0.1, help="set dropout rate of LSTMs")
parser.add_argument("--max-len", type=int, default=50, help="choose generated translation's max length")
parser.add_argument("--lr", type=float, default=1e-4, help="choose learning rate")
parser.add_argument("--epochs", type=int, default=15, help="choose epoch of training")
parser.add_argument("--mode", type=str, choices=['train', 'test'], help="choose train or test model")

args = parser.parse_args()

# set hyper parameters
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch
epochs = args.epochs
lr = args.lr
embed_size = args.embed_size
ffn_hid_size = args.ffn_hid_size
n_head = args.nhead
encoder_layer = args.encoder_layer
decoder_layer = args.decoder_layer
dropout = args.dropout
MAX_LENGTH = args.max_len

# prepare data
src_vocab_file = "../data/process_data/de.dict"
tgt_vocab_file = "../data/process_data/en.dict"
source_vocab = Vocab.load_from_file(src_vocab_file)
target_vocab = Vocab.load_from_file(tgt_vocab_file)


# for decoder predicting
def generate_square_subsequent_mask(sz):
    """
    :param sz: src_len
    :return: mask (src_len, src_len)
    # we can call transformer.generate_square_subsequent_mask
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# for encoder and decoder's padding tokens
def create_mask(src, tgt):
    """
    :param src: (src_len, batch)
    :param tgt: (tgt_len, batch)
    :return:
    """
    src_seq_len = src.size(0)
    tgt_seq_len = tgt.size(0)

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros(src_seq_len, src_seq_len, device=device, dtype=torch.bool)  # all false, nothing to mask

    src_padding_mask = (src == source_vocab.word2idx['<pad>']).transpose(0, 1)  # (batch, src_len)
    tgt_padding_mask = (tgt == target_vocab.word2idx['<pad>']).transpose(0, 1)  # (batch, src_len)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


train_file = "../data/process_data/train.txt"
valid_file = "../data/process_data/dev.txt"
test_file = "../data/process_data/test.txt"

train_dataset = MTDataset.load_from_file(train_file)
valid_dataset = MTDataset.load_from_file(valid_file)
test_dataset = MTDataset.load_from_file(test_file)


def collate(batch):
    src, tgt, = [], []
    for source, target in batch:
        src_tokens = torch.tensor(source, dtype=torch.long)
        src.append(src_tokens)
        tgt_tokens = torch.tensor([target_vocab.word2idx['<eos>']] + target + [target_vocab.word2idx['<eos>']], dtype=torch.long)
        tgt.append(tgt_tokens)

    src_tensor = pad_sequence(src, padding_value=source_vocab.word2idx['<pad>']).to(device)  # (src_len, batch)
    tgt_tensor = pad_sequence(tgt, padding_value=target_vocab.word2idx['<pad>']).to(device)  # (tgt_len, batch)
    return src_tensor, tgt_tensor


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate)
val_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate)


def train(model, criterion, train_loader, val_loader, learning_rate=0.01):
    min_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(1, 1 + epochs):
        model.train()
        train_loss = 0
        train_epoch = tqdm(train_loader, total=len(train_loader), leave=True)
        for batch in train_epoch:
            optimizer.zero_grad()
            # src, tgt: (length, batch)
            src, tgt = batch
            tgt_input = tgt[:-1, :]  # (length - 1, batch)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
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
                src, tgt = batch
                tgt_input = tgt[:-1, :]  # (length - 1, batch)
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                tgt_out = tgt[1:, :]

                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

                val_loss += loss.item()
                val_epoch.set_description(f"Validating")
                val_epoch.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch}/{epochs}]: Train loss: {train_loss},  Valid loss: {val_loss}")

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')


model = NMT(device, source_vocab, target_vocab, embed_size, n_head, encoder_layer, decoder_layer, ffn_hid_size,
            dropout=dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.get_token_by_word('<pad>'))
"""     
    1. layer=6  embed_size=512  ffn_hid_size=2048 lr=0.0001 MAX_LENGTH=50
         BLEU SCORE: BLEU = 19.47 52.6/24.6/13.6/8.2 (BP = 1.000 ratio = 1.005 hyp_len = 141753 ref_len = 141043)
"""
if args.mode == 'train':
    train(model, criterion, train_dataloader, val_dataloader, learning_rate=lr)

model.load_state_dict(torch.load('best_model.pt'))


def evaluate(model, test_loader):
    src, gen, ref = [], [], []
    model.eval()
    test_epoch = tqdm(test_loader, total=len(test_loader), leave=True)
    with torch.no_grad():
        for batch in test_epoch:
            src_tensor, tgt_tensor = batch
            src_mask = torch.zeros((src_tensor.size(0), src_tensor.size(0)), dtype=torch.bool, device=device)
            tokens = model.greedy_decode(src_tensor, src_mask, max_len=MAX_LENGTH)
            source_sentence = " ".join(source_vocab.convert_tokens_to_words(src_tensor.reshape(-1).tolist()))
            target_sentence = " ".join(chop_off_eos(target_vocab.convert_tokens_to_words(tgt_tensor.reshape(-1).tolist())))
            output_sentence = " ".join(chop_off_eos(target_vocab.convert_tokens_to_words(tokens.reshape(-1).tolist())))

            src.append(source_sentence)
            ref.append(target_sentence)
            gen.append(output_sentence)

    bleu = corpus_bleu(gen, [ref])
    print(f"BLEU SCORE: {bleu}")
    return src, ref, gen


test_src, test_ref, test_hyp = evaluate(model, test_dataloader)
random_check(test_src, test_ref, test_hyp, show=50)
