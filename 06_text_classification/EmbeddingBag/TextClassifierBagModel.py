"""
    预测结果
"""
import torch
from TextClassifier import TextClassifier
from torch.utils.data import Dataset, DataLoader
from DataProcess import DataProcess
from typing import List
from ClassificationModel import ClassificationModel
import pytorch_lightning as pl


# 实现用于文本数据集构建的map-style的DataPipe类
class DataPipe(Dataset):
    def __init__(self, text: List[str]):
        super(DataPipe, self).__init__()
        self.text = text
        self.datapipe = self.load_data()

    def load_data(self):
        datapipe = []
        for mes in self.text:
            datapipe.append((mes, '0'))
        return datapipe

    def __getitem__(self, item):
        return self.datapipe[item]

    def __len__(self):
        return len(self.datapipe)


class TextClassifierBiLSTM(TextClassifier):
    def __init__(self, train_path, model_path, embed_dim, num_class):
        super(TextClassifierBiLSTM, self).__init__()
        self.processer = DataProcess(train_path, val_path='', test_path='')  # 主要是为了获得其中的词汇表进行tensor转换
        self.model = ClassificationModel(vocab_size=len(self.processer.vocab), embed_size=_embed_dim, num_class=_num_class)
        self.trainer = pl.Trainer(accelerator='gpu', devices=1)
        self.model_path = model_path

    def classify_text(self, title: List[str], content: List[str]):
        data = DataPipe(title)
        dataloader = DataLoader(data, batch_size=len(title), shuffle=False, collate_fn=self.processer.collate_batch)
        predict_result = self.trainer.predict(self.model, dataloader, ckpt_path=self.model_path)
        return predict_result


if __name__ == '__main__':
    _embed_dim = 35
    _num_class = 2
    train_data_path = '../../shannon-bootcamp-data/06_text_classification/train_data_v3.json'
    model_param_path = 'checkpoint_model/batch=64-embed=35/epoch=2-train_loss=0.21-train_F1=0.92-val_loss=0.32-val_acc=0.86-val_precision=0.81-val_recall=0.90-val_F1=0.85.ckpt'
    classifier = TextClassifierBiLSTM(train_data_path, model_param_path, _embed_dim, _num_class)

    test_titles = [
        '51信用卡CEO孙海涛：“科技+金融”催生金融新世界',
        '美要求对中追加1000亿美元关税(附声明全文)',
        '敲除病虫害基因 让棉花高产又“绿色”',
        '西气东输三线长沙支线工程完工 主要承担向长沙(湘江西)、益阳、常德等地的供气任务',
        '个税变化：元旦后发年终奖 应纳税额有的“打三折”'
    ]
    classify_result = classifier.classify_text(test_titles, [])[0]  # 真实的结果为[0,1,0,0,1]
    print(classify_result)
