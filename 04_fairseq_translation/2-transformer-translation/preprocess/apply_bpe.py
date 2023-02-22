"""
  * FileName: apply_bpe.py
  * Author:   Slatter
  * Date:     2023/2/22 15:44
  * Description:  
"""
import json
import fastBPE
from tqdm import tqdm


def run():
    codes_path = 'codes'
    src_vocab_path = 'vocab.src.40000'
    tgt_vocab_path = 'vocab.tgt.40000'

    src_bpe = fastBPE.fastBPE(codes_path, src_vocab_path)
    tgt_bpe = fastBPE.fastBPE(codes_path, tgt_vocab_path)

    for split in ['train', 'valid', 'test']:
        with open(f'../dataset/tokenized/{split}.json', 'r') as output_file:
            data = json.load(output_file)

        bpe_data = []
        for pair in tqdm(data, desc=split):
            bpe_data.append({
                'src': src_bpe.apply([" ".join(pair['src'])])[0].split(' '),
                'tgt': tgt_bpe.apply([" ".join(pair['tgt'])])[0].split(' ')
            })

        with open(f"../dataset/processed_data/{split}.json", "w") as output_file:
            json.dump(bpe_data, output_file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    run()
