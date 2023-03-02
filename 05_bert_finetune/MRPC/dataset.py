"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/2/26 16:25
  * Description:  
"""
from torch.utils.data import Dataset


class MRPCDataset(Dataset):
    def __init__(self, pairs):
        super(MRPCDataset, self).__init__()
        self.pairs = pairs

    @classmethod
    def load_dataset(cls, split: str, path: str):
        pairs = []
        if split == 'train' or split == 'dev':
            positive, negative = [], []
            with open(path, 'r', encoding='utf-8') as input_file:
                cnt = 0
                for line in input_file:
                    cnt += 1
                    if cnt == 1:
                        continue
                    piece = line.strip().split('\t')
                    if int(piece[0]) == 0:
                        negative.append((piece[3], piece[4], int(piece[0])))
                    else:
                        positive.append((piece[3], piece[4], int(piece[0])))

            if split == 'train':
                pairs = positive[:int(0.9 * len(positive))] + negative[:int(0.9 * len(negative))]
            else:
                pairs = positive[int(0.9 * len(positive)):] + negative[int(0.9 * len(negative)):]
        else:
            with open(path, 'r', encoding='utf-8') as input_file:
                cnt = 0
                for line in input_file:
                    cnt += 1
                    if cnt == 1:
                        continue
                    piece = line.strip().split('\t')
                    pairs.append((piece[3], piece[4], int(piece[0])))
        return cls(pairs)

    def __getitem__(self, idx):
        text1, text2, label = self.pairs[idx]
        return text1, text2, label

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    train_path = '/data2/daijincheng/datasets/GLUE/MSRParaphrase/msr_paraphrase_train.txt'
    train_dataset = MRPCDataset.load_dataset('train', train_path)
    print(len(train_dataset))
    print(train_dataset[0])

    dev_path = '/data2/daijincheng/datasets/GLUE/MSRParaphrase/msr_paraphrase_train.txt'
    dev_dataset = MRPCDataset.load_dataset('dev', dev_path)
    print(len(dev_dataset))
    print(dev_dataset[0])

    test_path = '/data2/daijincheng/datasets/GLUE/MSRParaphrase/msr_paraphrase_test.txt'
    test_dataset = MRPCDataset.load_dataset('test', test_path)
    print(len(test_dataset))
    print(test_dataset[0])
