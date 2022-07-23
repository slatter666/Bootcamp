from torch.utils.data import Dataset
import json
from typing import Tuple


# 实现用于文本数据集构建的map-style的TextDataset类
class TextDataset(Dataset):
    def __init__(self, path):
        super(TextDataset, self).__init__()
        self.path = path
        self.datapipe = self.load_data()

    def load_data(self):
        datapipe = []
        with open(self.path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
            for mes in data:
                datapipe.append((mes['text'], mes['label']))
        return datapipe

    def __getitem__(self, item) -> Tuple:
        return self.datapipe[item]

    def __len__(self):
        return len(self.datapipe)


if __name__ == '__main__':
    path = '../../shannon-bootcamp-data/06_text_classification/train_data_v3.json'
    ins = TextDataset(path)

