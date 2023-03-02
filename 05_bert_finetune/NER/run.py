"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/2/26 17:12
  * Description:
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer
from functools import partial
from dataset import NERDataset
from model import NERBert

transformers.logging.set_verbosity_error()
torch.set_float32_matmul_precision('medium')

# hyper parameters
batch_size = 32
num_class = 9
epochs = 4
lr = 5e-5
save_dir = 'checkpoints'

tokenizer = BertTokenizer.from_pretrained('bert_finetune-base-cased')


def collate(batch, tokenizer):
    cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    input_ids, attention_mask, lengths, label = [], [], [], []
    for tokens, target in batch:
        input_ids.append([cls] + tokenizer.convert_tokens_to_ids(tokens) + [sep])
        lengths.append(len(tokens))
        attention_mask.append([1] * (len(tokens) + 2))
        label.append(target)

    max_len = max(lengths)  # max(lengths)是最大token数
    label = [x + [-1] * (max_len - len(x)) for x in label]  # padding label  (batch, max_len)

    input_ids = torch.tensor([x + [pad] * (max_len + 2 - len(x)) for x in input_ids])  # 加2是需要算[CLS]和[SEP]
    attention_mask = torch.tensor([x + [0] * (max_len + 2 - len(x)) for x in attention_mask])
    token_type_ids = torch.zeros_like(input_ids)

    enc_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
    label = torch.LongTensor(label)
    return enc_inputs, label


def train():
    train_path = '/data2/daijincheng/datasets/CoNLL-2003/processed/train.json'
    dev_path = '/data2/daijincheng/datasets/CoNLL-2003/processed/valid.json'

    train_dataset = NERDataset.load_dataset(train_path)
    dev_dataset = NERDataset.load_dataset(dev_path)
    print(f'Train samples: {len(train_dataset)}, valid samples: {len(dev_dataset)}')

    collate_fn = partial(collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)

    model = NERBert(num_class, lr)

    # training
    checkpoint_path = os.path.join(save_dir, str(lr))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=checkpoint_path,
        filename='{epoch}-{val_f1:.2f}',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, dev_loader)


def test(ck_path: str):
    test_path = '/data2/daijincheng/datasets/CoNLL-2003/processed/test.json'
    test_dataset = NERDataset.load_dataset(test_path)
    print(f'Test samples: {len(test_dataset)}')

    collate_fn = partial(collate, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128, collate_fn=collate_fn, num_workers=8)

    model = NERBert(num_class, lr)

    trainer = pl.Trainer(accelerator='gpu', devices=1)
    trainer.test(model, test_loader, ckpt_path=ck_path)


if __name__ == '__main__':
    # train()
    ck_path = './checkpoints/2e-05/epoch=2-val_f1=0.98.ckpt'
    test(ck_path)
