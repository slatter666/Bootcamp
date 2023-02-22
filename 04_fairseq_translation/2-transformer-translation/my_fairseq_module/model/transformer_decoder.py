"""
  * FileName: transformer_decoder.py
  * Author:   Slatter
  * Date:     2023/2/18 14:51
  * Description:  
"""
from fairseq.models.transformer import TransformerDecoder


class Decoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
