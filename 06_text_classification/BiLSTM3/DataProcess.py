import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import json
from TextDataset import TextDataset


# 实现用于文本处理的DataProcess类
class DataProcess():
    def __init__(self, train_path, val_path, test_path, pad_token='<pad>', batch_size=8, device=torch.device('cuda')):
        super(DataProcess, self).__init__()
        # Hyper parameters
        self.BATCH_SIZE = batch_size
        self.DEVICE = device

        # local parameters
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.pad_token = pad_token
        self.max_len = -1
        self.vocab = self.get_vocab()

        # pipeline
        self.text_pipeline = lambda x: self.vocab(x)
        self.label_pipeline = lambda x: int(x)

    def yield_tokens(self):
        with open(self.train_path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
            for mes in data:
                text = mes['text']
                cut = list(text)
                self.max_len = max(self.max_len, len(cut))
                yield cut

    def get_vocab(self):
        """
        :return: 根据训练集构建好的词汇表
        """
        vocab = build_vocab_from_iterator(self.yield_tokens(), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        vocab.append_token(self.pad_token)
        return vocab

    def collate_batch(self, batch):
        label_list, text_list, length_list = [], [], []
        for (text, label) in batch:
            label_list.append(self.label_pipeline(label))
            cut_text = list(text)
            length_list.append(min(len(cut_text), self.max_len))
            if len(cut_text) > self.max_len:   # 如果验证集或者测试集出现了句子包含词数超过训练集最长词数，那么采取截断方式
                full_text = cut_text[:self.max_len]
            else:
                full_text = cut_text + [self.pad_token] * (self.max_len - len(cut_text))
            processed_text = self.text_pipeline(full_text)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.tensor(text_list, dtype=torch.int64)
        return label_list.to(self.DEVICE), text_list.to(self.DEVICE), length_list

    def get_dataloader(self):
        """
            return: 训练集、验证集、测试集的Dataloader
        """
        train_dataset = TextDataset(self.train_path)
        val_dataset = TextDataset(self.val_path)
        test_dataset = TextDataset(self.test_path)

        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True,
                                      collate_fn=self.collate_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=False,
                                    collate_fn=self.collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False,
                                     collate_fn=self.collate_batch)
        return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_data_path = '../../shannon-bootcamp-data/06_text_classification/train_data_v3.json'
    val_data_path = '../../shannon-bootcamp-data/06_text_classification/valid_data_v3.json'
    test_data_path = '../../shannon-bootcamp-data/06_text_classification/test_data_v3.json'
    ins = DataProcess(train_data_path, val_data_path, test_data_path)
