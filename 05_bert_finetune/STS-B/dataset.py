"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/2/26 16:25
  * Description:  
"""
import pandas as pd
from torch.utils.data import Dataset


class STSDataset(Dataset):
    def __init__(self, pairs):
        super(STSDataset, self).__init__()
        self.pairs = pairs

    @classmethod
    def load_dataset(cls, split: str, path: str):
        pairs = []
        if split == 'train' or split == 'dev':
            with open(path, 'r', encoding='utf-8') as input_file:
                cnt = 0
                for line in input_file:
                    cnt += 1
                    if cnt == 1:
                        continue
                    piece = line.strip().split('\t')
                    pairs.append((piece[-3], piece[-2], float(piece[-1])))
        else:
            with open(path, 'r', encoding='utf-8') as input_file:
                cnt = 0
                for line in input_file:
                    cnt += 1
                    if cnt == 1:
                        continue
                    piece = line.strip().split('\t')
                    pairs.append((piece[-2], piece[-1], int(piece[0])))
        return cls(pairs)

    def __getitem__(self, idx):
        text1, text2, label = self.pairs[idx]
        return text1, text2, label

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    train_path = '/data2/daijincheng/datasets/GLUE/STS-B/train.tsv'
    train_dataset = STSDataset.load_dataset('train', train_path)
    print(len(train_dataset))
    print(train_dataset[0])

    dev_path = '/data2/daijincheng/datasets/GLUE/STS-B/dev.tsv'
    dev_dataset = STSDataset.load_dataset('dev', dev_path)
    print(len(dev_dataset))
    print(dev_dataset[0])

    test_path = '/data2/daijincheng/datasets/GLUE/STS-B/test.tsv'
    test_dataset = STSDataset.load_dataset('test', test_path)
    print(len(test_dataset))
    print(test_dataset[0])
