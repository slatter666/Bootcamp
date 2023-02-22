"""
  * FileName: build_dict.py
  * Author:   Slatter
  * Date:     2023/2/22 09:27
  * Description:  
"""
import json
from collections import Counter

from fairseq.data import Dictionary


def save_dict(src_dict: Dictionary, tgt_dict: Dictionary, base_dir: Dictionary):
    src_dict.save(f"{base_dir}/dict.src.txt")
    tgt_dict.save(f"{base_dir}/dict.tgt.txt")


def build_dictionary(file_name, key):
    with open(file_name) as f:
        data = json.load(f)

    dict = Dictionary()
    counter = Counter()

    for pair in data:
        counter.update(pair[key])
        counter.update([dict.eos_word])

    for word, cnt in sorted(counter.items()):
        dict.add_symbol(word, cnt)
    dict.finalize()
    return dict


def run():
    base_dir = '../dataset/processed_data'
    src_dict = build_dictionary(f'{base_dir}/train.json', 'src')
    tgt_dict = build_dictionary(f'{base_dir}/train.json', 'tgt')
    save_dict(src_dict, tgt_dict, base_dir)


if __name__ == '__main__':
    run()
