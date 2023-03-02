"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/2/26 16:25
  * Description:  
"""
import json
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, pairs):
        super(NERDataset, self).__init__()
        self.mapping = {'B-ORG': 0, 'I-ORG': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4, 'O': 5, 'I-MISC': 6, 'B-MISC': 7, 'B-LOC': 8}
        self.pairs = pairs

    @classmethod
    def load_dataset(cls, path: str):
        with open(path, 'r', encoding='utf-8') as input_file:
            pairs = json.load(input_file)
        return cls(pairs)

    def __getitem__(self, idx):
        tokens, tags = self.pairs[idx]['tokens'], self.pairs[idx]['tags']
        tags = [self.mapping[x] for x in tags]
        return tokens, tags

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    train_path = '/data2/daijincheng/datasets/CoNLL-2003/processed/train.json'
    train_dataset = NERDataset.load_dataset(train_path)
    print(len(train_dataset))
    print(train_dataset[0])

    dev_path = '/data2/daijincheng/datasets/CoNLL-2003/processed/valid.json'
    dev_dataset = NERDataset.load_dataset(dev_path)
    print(len(dev_dataset))
    print(dev_dataset[0])

    test_path = '/data2/daijincheng/datasets/CoNLL-2003/processed/test.json'
    test_dataset = NERDataset.load_dataset(test_path)
    print(len(test_dataset))
    print(test_dataset[0])
