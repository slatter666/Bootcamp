"""
  * FileName: application.py
  * Author:   Slatter
  * Date:     2023/2/23 10:57
  * Description:  
"""
import fastBPE
from fairseq.hub_utils import GeneratorHubInterface

from my_fairseq_module.model import NMT


def run():
    model: GeneratorHubInterface = NMT.from_pretrained(
        model_name_or_path="checkpoints",
        checkpoint_file="checkpoint_best.pt",
        data_name_or_path="dataset/processed_data",
        tokenizer=None,
    )

    codes_path = 'preprocess/codes'
    src_vocab_path = 'preprocess/vocab.src.40000'
    tgt_vocab_path = 'preprocess/vocab.tgt.40000'
    src_bpe = fastBPE.fastBPE(codes_path, src_vocab_path)
    tgt_bpe = fastBPE.fastBPE(codes_path, tgt_vocab_path)
    bpe_symbol = "@@ "
    while True:
        sentence = input('\nInput: ')
        translation = model.translate([src_bpe.apply([sentence])[0]])  # 注意这里是特地写的繁琐的，为了强调二者接收的都是list，而不是单个句子
        print(translation)
        # 下面这一行模仿了fairseq例fastBPE的处理方法，参见 https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/encoders/fastbpe.py
        print((translation[0] + " ").replace(bpe_symbol, "").rstrip())


if __name__ == '__main__':
    run()
