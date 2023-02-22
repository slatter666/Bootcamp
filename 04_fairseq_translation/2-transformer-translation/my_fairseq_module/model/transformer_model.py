"""
  * FileName: transformer_model.py
  * Author:   Slatter
  * Date:     2023/2/18 14:53
  * Description:  
"""
from fairseq.models import register_model
from fairseq.models.transformer import TransformerModel

from .transformer_encoder import Encoder
from .transformer_decoder import Decoder


@register_model('simple_transformer')
class NMT(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return Encoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return Decoder(args, tgt_dict, embed_tokens)
