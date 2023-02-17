"""
  * FileName: lstm_model.py
  * Author:   Slatter
  * Date:     2022/10/14 21:58
  * Description:  
"""
from fairseq.models import FairseqEncoderDecoderModel, register_model
from .lstm_encoder import LSTMEncoder
from .lstm_decoder import LSTMDecoder


@register_model("lstm_model")
class LSTMModel(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the encoder embeddings',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--num-layer', type=int, metavar='N',
            help='layer of the architecture',
        )
        parser.add_argument(
            '--dropout', type=float, help='dropout rate'
        )

    @classmethod
    def build_model(cls, args, task):
        encoder = LSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            layer=args.num_layer,
            dropout=args.dropout
        )
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            layer=args.num_layer,
            dropout=args.dropout
        )
        model = LSTMModel(encoder, decoder)
        print(model)
        return model
