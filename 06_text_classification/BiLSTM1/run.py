"""
  * FileName: run
  * Author:   Slatter
  * Date:     2022/7/21 17:34
  * Description: 实现模型BiLSTMModel1的运行
  * History:
  * <author>          <time>          <version>          <desc>
  * 作者姓名           修改时间           版本号              描述
"""

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from BiLSTMModel1 import BiLSTMModel1
from DataProcess import DataProcess

if __name__ == '__main__':
    # hyper parameters
    _batch_size = 128
    _embed_dim = 500
    _hidden_dim = 120
    _num_layers = 3
    _num_class = 2
    _epochs = 20
    _save_dir = 'checkpoint_model'

    # data
    train_data_path = '../../shannon-bootcamp-data/06_text_classification/train_data_v3.json'
    val_data_path = '../../shannon-bootcamp-data/06_text_classification/valid_data_v3.json'
    test_data_path = '../../shannon-bootcamp-data/06_text_classification/test_data_v3.json'
    processer = DataProcess(train_data_path, val_data_path, test_data_path, batch_size=_batch_size)
    train_dataloder, val_dataloader, test_dataloader = processer.get_dataloader()

    # model
    model = BiLSTMModel1(vocab_size=len(processer.vocab), embed_size=_embed_dim, hidden_size=_hidden_dim,
                        num_layers=_num_layers, num_class=_num_class)

    # training
    checkpoint_path = _save_dir + '/batch={}-embed={}-hidden={}-layer={}'.format(_batch_size, _embed_dim, _hidden_dim, _num_layers)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_F1',
        dirpath=checkpoint_path,
        filename='{epoch}-{train_loss:.2f}-{train_F1:.2f}-{val_loss:.2f}-{val_acc:.2f}-{val_precision:.2f}-{val_recall:.2f}-{val_F1:.2f}',
        save_top_k=5,
        mode='min',
        save_weights_only=True
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=_epochs, callbacks=[checkpoint_callback],
                         log_every_n_steps=10)
    # trainer.fit(model, train_dataloder, val_dataloader)

    for path in os.listdir(checkpoint_path):
        file_path = checkpoint_path + '/' + path
        trainer.test(model, test_dataloader, ckpt_path=file_path)

"""
---------------------------验证集---------------------------
SVM模型在验证集上准确率为: 0.902542372881356
SVM模型在验证集上查准率为: 0.8931818181818182
SVM模型在验证集上查全率为: 0.8972602739726028
SVM模型在验证集上F1值为: 0.8952164009111616
---------------------------测试集---------------------------
SVM模型在测试集上准确率为: 0.8380952380952381
SVM模型在测试集上查准率为: 0.9136125654450262
SVM模型在测试集上查全率为: 0.744136460554371
SVM模型在测试集上F1值为: 0.8202115158636898
[0.0, 1.0, 0.0, 0.0, 0.0]
"""
