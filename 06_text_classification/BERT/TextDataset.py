import json
from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, path, tokenizer):
        super(TextDataset, self).__init__()
        self.maxlen = 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            texts = [mes['text'] for mes in data]
            labels = torch.tensor([int(mes['label']) for mes in data])
            tokenized_text = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

        self.data = [{"input_ids": tokenized_text.input_ids[i], "attention_mask": tokenized_text.attention_mask[i], "labels": labels[i]} for i in range(len(texts))]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
