"""
  * FileName: transformer_encoder.py
  * Author:   Slatter
  * Date:     2023/2/18 14:43
  * Description:  
"""
from fairseq.models.transformer import TransformerEncoder


class Encoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        super().__init__(args, dictionary, embed_tokens, return_fc)
