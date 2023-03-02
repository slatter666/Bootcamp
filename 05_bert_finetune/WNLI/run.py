"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/2/26 17:12
  * Description:  
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer
from functools import partial
from dataset import WNLIDataset
from model import WNLIBert

transformers.logging.set_verbosity_error()
torch.set_float32_matmul_precision('medium')

# hyper parameters
batch_size = 32
num_class = 2
epochs = 4
lr = 2e-5
save_dir = 'checkpoints'

tokenizer = BertTokenizer.from_pretrained('bert_finetune-base-uncased')


def collate(batch, tokenizer):
    sent1, sent2, label = [], [], []
    for text1, text2, target in batch:
        sent1.append(text1)
        sent2.append(text2)
        label.append(target)

    enc_inputs = tokenizer(sent1, sent2, return_tensors='pt', padding=True, truncation=True)
    label = torch.LongTensor(label)
    return enc_inputs, label


def train():
    train_path = '/data2/daijincheng/datasets/GLUE/WNLI/train.tsv'
    dev_path = '/data2/daijincheng/datasets/GLUE/WNLI/dev.tsv'

    train_dataset = WNLIDataset.load_dataset('train', train_path)
    dev_dataset = WNLIDataset.load_dataset('dev', dev_path)
    print(f'Train samples: {len(train_dataset)}, valid samples: {len(dev_dataset)}')

    collate_fn = partial(collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)

    model = WNLIBert(num_class, lr)

    # training
    checkpoint_path = os.path.join(save_dir, str(lr))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=checkpoint_path,
        filename='{epoch}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, dev_loader)


def predict(ck_path, save_path):
    test_path = '/data2/daijincheng/datasets/GLUE/WNLI/test.tsv'
    test_dataset = WNLIDataset.load_dataset('test', test_path)
    print(f'Test samples: {len(test_dataset)}')

    collate_fn = partial(collate, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)

    model = WNLIBert(num_class, lr)

    trainer = pl.Trainer(accelerator='gpu', devices=1)
    prediction = trainer.predict(model, test_loader, ckpt_path=ck_path)

    res, indexes = [], []
    for pred, index in prediction:
        res += pred
        indexes += index

    with open(save_path, 'w', encoding='utf-8') as output_file:
        output_file.write('index \t prediction\n')
        for idx, pred in zip(indexes, res):
            output_file.write(f'{idx} \t {pred}\n')


if __name__ == '__main__':
    # train()
    ck_path = './checkpoints/4e-05/epoch=0-val_acc=0.56.ckpt'
    save_path = '/data2/daijincheng/bert_finetune/submission/WNLI.tsv'
    predict(ck_path, save_path)
