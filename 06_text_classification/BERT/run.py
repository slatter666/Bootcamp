import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from BERTModel import BERTModel
from TextDataset import TextDataset

if __name__ == '__main__':
    # hyper parameters
    _batch_size = 20  # 32撑不太住, 估计24对于我的4G显存已经极限了
    _num_class = 2
    _epochs = 5
    _save_dir = 'checkpoints'

    # 准备数据集
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    train_data_path = '../../shannon-bootcamp-data/06_text_classification/train_data_v3.json'
    val_data_path = '../../shannon-bootcamp-data/06_text_classification/valid_data_v3.json'
    test_data_path = '../../shannon-bootcamp-data/06_text_classification/test_data_v3.json'

    train_dataset = TextDataset(train_data_path, tokenizer)
    val_dataset = TextDataset(val_data_path, tokenizer)
    test_dataset = TextDataset(test_data_path, tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=_batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=_batch_size)

    # model
    model = BERTModel(_num_class)

    # training
    checkpoint_path = _save_dir + '/batch={}'.format(_batch_size)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_F1',
        dirpath=checkpoint_path,
        filename='{epoch}-{train_loss:.2f}-{train_F1:.2f}-{val_loss:.2f}-{val_acc:.2f}-{val_precision:.2f}-{val_recall:.2f}-{val_F1:.2f}',
        save_top_k=3,
        mode='min',
        save_weights_only=True
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=_epochs, callbacks=[checkpoint_callback],
                         log_every_n_steps=20)
    trainer.fit(model, train_dataloader, val_dataloader)

    for path in os.listdir(checkpoint_path):
        file_path = checkpoint_path + '/' + path
        trainer.test(model, test_dataloader, ckpt_path=file_path)
