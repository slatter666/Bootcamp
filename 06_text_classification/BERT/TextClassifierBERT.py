"""
    预测结果
"""
from typing import List
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from BERTModel import BERTModel
from TextClassifier import TextClassifier


class TextClassifierBERT(TextClassifier):
    def __init__(self, model_path, num_class):
        super(TextClassifierBERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model = BERTModel(num_class=num_class)
        self.trainer = pl.Trainer(accelerator='gpu', devices=1)
        self.model_path = model_path

    def classify_text(self, title: List[str], content: List[str]):
        tokenized_title = self.tokenizer(title, return_tensors='pt', padding=True, truncation=True)
        data = [{"input_ids": tokenized_title.input_ids[i], "attention_mask": tokenized_title.attention_mask[i], "labels": torch.tensor(0)} for i in range(len(title))]  # labels就直接造个0
        dataloader = DataLoader(data, batch_size=16)
        predict_result = self.trainer.predict(self.model, dataloader, ckpt_path=self.model_path)
        return predict_result


if __name__ == '__main__':
    _num_class = 2
    model_param_path = "checkpoints/batch=20/epoch=2-train_loss=0.09-train_F1=1.00-val_loss=0.41-val_acc=0.90-val_precision=0.82-val_recall=1.00-val_F1=0.90.ckpt"
    classifier = TextClassifierBERT(model_param_path, _num_class)

    test_titles = [
        '51信用卡CEO孙海涛：“科技+金融”催生金融新世界',
        '美要求对中追加1000亿美元关税(附声明全文)',
        '敲除病虫害基因 让棉花高产又“绿色”',
        '西气东输三线长沙支线工程完工 主要承担向长沙(湘江西)、益阳、常德等地的供气任务',
        '个税变化：元旦后发年终奖 应纳税额有的“打三折”'
    ]
    classify_result = classifier.classify_text(test_titles, [])[0]  # 真实的结果为[0,1,0,0,1]
    print(classify_result)
