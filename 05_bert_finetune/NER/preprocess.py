"""
  * FileName: preprocess.py
  * Author:   Slatter
  * Date:     2023/3/1 10:42
  * Description:  Preprocess CoNLL-2003 Dataset,
                  you can get the raw data from https://github.com/ningshixian/NER-CONLL2003/tree/master/data,
                  thanks for dataset provider
"""
import json
import os


def process(read_dir: str, write_dir: str, split: str):
    """
    preprocess raw data, remember in the processing we include starter like '-DOCSTART- -X- -X- O',
    so that our dataset can match it to the original data set
    :param read_dir: raw direcotry
    :param write_dir: processed directory
    :param split: train, valid, test
    :return:
    """
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    read_path = os.path.join(read_dir, f'{split}.txt')
    write_path = os.path.join(write_dir, f'{split}.json')

    sentences, tokens, tags = [], [], []
    with open(read_path, 'r') as input_file:
        for line in input_file:
            line = line.strip()

            if line == '':
                if len(tokens) != 0:
                    sentences.append({'tokens': tokens, 'tags': tags})
                    tokens, tags = [], []
            else:
                contents = line.split()
                tokens.append((contents[0]))
                tags.append((contents[-1]))

    if len(tokens) != 0:
        sentences.append({'tokens': tokens, 'tags': tags})

    print(f"{split} samples: {len(sentences)}")
    with open(write_path, 'w') as output_file:
        json.dump(sentences, output_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    read_dir = '/data2/daijincheng/datasets/CoNLL-2003/raw'
    write_dir = '/data2/daijincheng/datasets/CoNLL-2003/processed'
    process(read_dir, write_dir, 'train')
    process(read_dir, write_dir, 'valid')
    process(read_dir, write_dir, 'test')
