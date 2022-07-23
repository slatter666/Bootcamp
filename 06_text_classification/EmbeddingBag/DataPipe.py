"""
  * FileName: DataPipe
  * Author:   Slatter
  * Date:     2022/7/22 19:53
  * Description: 
  * History:
  * <author>          <time>          <version>          <desc>
  * 作者姓名           修改时间           版本号              描述
"""
import json
from typing import Tuple
from torch.utils.data import Dataset


class DataPipe(Dataset):
    def __init__(self, path):
        super(DataPipe, self).__init__()
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
