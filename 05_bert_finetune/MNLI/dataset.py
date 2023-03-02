"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/2/26 16:25
  * Description:  
"""
import pandas as pd
from torch.utils.data import Dataset


class MNLIDataset(Dataset):
    def __init__(self, pairs):
        super(MNLIDataset, self).__init__()
        self.mapping = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
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
                    pairs.append((piece[8], piece[9], piece[-1]))
        else:
            with open(path, 'r', encoding='utf-8') as input_file:
                cnt = 0
                for line in input_file:
                    cnt += 1
                    if cnt == 1:
                        continue
                    piece = line.strip().split('\t')
                    pairs.append((piece[8], piece[9], int(piece[0])))
        return cls(pairs)

    def __getitem__(self, idx):
        text1, text2, label = self.pairs[idx]
        if isinstance(label, str):
            label = self.mapping[label]
        return text1, text2, label

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    train_path = '/data2/daijincheng/datasets/GLUE/MNLI/train.tsv'
    train_dataset = MNLIDataset.load_dataset('train', train_path)
    print(len(train_dataset))
    print(train_dataset[0])

    dev_path = '/data2/daijincheng/datasets/GLUE/MNLI/dev_matched.tsv'
    dev_dataset = MNLIDataset.load_dataset('dev', dev_path)
    print(len(dev_dataset))
    print(dev_dataset[0])

    test_path = '/data2/daijincheng/datasets/GLUE/MNLI/test_mismatched.tsv'
    test_dataset = MNLIDataset.load_dataset('test', test_path)
    print(len(test_dataset))
    print(test_dataset[0])
