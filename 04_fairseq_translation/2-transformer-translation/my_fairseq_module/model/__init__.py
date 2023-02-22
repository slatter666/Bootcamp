"""
  * FileName: __init__.py.py
  * Author:   Slatter
  * Date:     2023/2/18 14:41
  * Description:  
"""
from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture

from .transformer_model import NMT


@register_model_architecture("simple_transformer", "nmt")
def build_model(args):
    base_architecture(args)
