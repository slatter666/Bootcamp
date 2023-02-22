"""
  * FileName: tokenize_data.py.py
  * Author:   Slatter
  * Date:     2023/2/22 14:35
  * Description:  
"""
import json

from sacremoses import MosesTokenizer
from tqdm import tqdm


def run():
    de_tokenizer = MosesTokenizer(lang='de')
    en_tokenizer = MosesTokenizer(lang='en')
    for split in ['train', 'valid', 'test']:
        with open(f'../dataset/raw_data/{split}.de', 'r') as input_file:
            source_list = [line.strip() for line in input_file]

        with open(f'../dataset/raw_data/{split}.en', 'r') as input_file:
            target_list = [line.strip() for line in input_file]

        data = []
        for src, tgt in tqdm(zip(source_list, target_list), total=len(source_list), desc=f"Tokenize {split}"):
            data.append({
                "src": de_tokenizer.tokenize(src, return_str=False),
                "tgt": en_tokenizer.tokenize(tgt, return_str=False),
            })

        with open(f'../dataset/tokenized/{split}.json', 'w') as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=2)

        # 给fastbpe学习bpe
        if split == "train":
            src = [" ".join(datum["src"]) for datum in data]
            tgt = [" ".join(datum["tgt"]) for datum in data]
            with open(f"../dataset/tokenized/train.for_bpe.src", "w") as output_file:
                output_file.write("\n".join(src))
            with open(f"../dataset/tokenized/train.for_bpe.tgt", "w") as output_file:
                output_file.write("\n".join(tgt))


if __name__ == '__main__':
    run()
