"""
  * FileName: __init__.py.py
  * Author:   Slatter
  * Date:     2022/10/11 20:18
  * Description:  
"""
from fairseq.models import register_model_architecture
from .lstm_model import LSTMModel


@register_model_architecture("lstm_model", "ende_translate")
def ende_translate(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
    args.layer = getattr(args, 'num_layer', 1)
    args.dropout = getattr(args, 'dropout', 0.1)
