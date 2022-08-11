import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import optim
from TextDataset import TextDataset


class BERTModel(pl.LightningModule):
    def __init__(self, num_class=2, LR=2e-5):
        super(BERTModel, self).__init__()
        # hyper parameters
        self.LR = LR

        self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_class)

    def forward(self, input_ids, attention_mask, labels):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.LR)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # {"input_ids":[], "attention_mask":[], "labels":[]}
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self.forward(input_ids, attention_mask, labels)
        logits = outputs.logits
        train_loss = outputs.loss

        train_acc, train_precision, train_recall, train_F1 = self.compute_metrics(logits, labels)
        self.log("train_loss", train_loss.item())
        self.log("train_acc", train_acc)
        self.log("train_precision", train_precision)
        self.log("train_recall", train_recall)
        self.log("train_F1", train_F1)
        self.print("train_loss:{:.2f} | train_acc:{:.2f} | train_precison:{:.2f} | train_recall:{:.2f} | train_F1:{"
                   ":.2f}".format(train_loss.item(), train_acc, train_precision, train_recall, train_F1))
        return train_loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self.forward(input_ids, attention_mask, labels)
        logits = outputs.logits
        val_loss = outputs.loss

        val_acc, val_precision, val_recall, val_F1 = self.compute_metrics(logits, labels)
        self.log("val_loss", val_loss.item())
        self.log("val_acc", val_acc)
        self.log("val_precision", val_precision)
        self.log("val_recall", val_recall)
        self.log("val_F1", val_F1)
        self.print("val_loss:{:.2f} | val_acc:{:.2f} | val_precison:{:.2f} | val_recall:{:.2f} | val_F1:{"
                   ":.2f}".format(val_loss.item(), val_acc, val_precision, val_recall, val_F1))

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self.forward(input_ids, attention_mask, labels)
        logits = outputs.logits
        test_loss = outputs.loss

        test_acc, test_precision, test_recall, test_F1 = self.compute_metrics(logits, labels)
        self.log("test_loss", test_loss.item())
        self.log("test_acc", test_acc)
        self.log("test_precision", test_precision)
        self.log("test_recall", test_recall)
        self.log("test_F1", test_F1)
        self.print("test_loss:{:.2f} | test_acc:{:.2f} | test_precison:{:.2f} | test_recall:{:.2f} | test_F1:{"
                   ":.2f}".format(test_loss.item(), test_acc, test_precision, test_recall, test_F1))

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self.forward(input_ids, attention_mask, labels)
        logits = outputs.logits
        predict = logits.argmax(dim=-1)  # 首先计算预测值
        return predict.tolist()

    def compute_metrics(self, logits, labels):
        """
            :param logits: 经过前向传播得到的logits with shape (b, num_class)
            :param ground_truth: 真实值
            :return: 准确率、查准率、查全率、F1  (acc、precision、recall、F1)
        """
        predict = logits.argmax(dim=-1)  # 首先计算预测值

        # 计算准确率
        correct = (predict == labels).sum()
        total = predict.size(0)
        acc = (correct / total).item()

        # 计算查全率、查准率、F1
        TP = ((predict == 1) & (labels == 1)).sum().item()
        FP = ((predict == 1) & (labels == 0)).sum().item()
        TN = ((predict == 0) & (labels == 0)).sum().item()
        FN = ((predict == 0) & (labels == 1)).sum().item()
        precision = TP / (TP + FP) if TP != 0 else 0
        recall = TP / (TP + FN) if TP != 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if precision != 0 and recall != 0 else 0
        return acc, precision, recall, F1


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    train_data_path = '../../shannon-bootcamp-data/06_text_classification/train_data_v3.json'
    val_data_path = '../../shannon-bootcamp-data/06_text_classification/valid_data_v3.json'
    test_data_path = '../../shannon-bootcamp-data/06_text_classification/test_data_v3.json'

    train_dataset = TextDataset(train_data_path, tokenizer)
    val_dataset = TextDataset(val_data_path, tokenizer)
    test_dataset = TextDataset(test_data_path, tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=16)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=16)

    model = BERTModel(num_class=2)
    for idx, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        print("batch:", batch)
        result = model(input_ids, attention_mask, labels)
        print("model(batch):", result)
        break

    # 设置训练参数
    # arguments = TrainingArguments(
    #     output_dir="checkpoints",
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     num_train_epochs=3,
    #     evaluation_strategy="epoch",  # run validation at the end of each epoch
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     load_best_model_at_end=True,
    #     seed=123
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=arguments,
    #     train_dataset=train_dataloader,
    #     eval_dataset=test_dataloader,  # change to test when you do your final evaluation!
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )
    #
    # trainer.train()
