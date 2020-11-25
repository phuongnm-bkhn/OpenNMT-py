"""Module defining decoders."""
from onmt.decoders.combined_transformer_rnn import CombinedTransformerRnnDecoder
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder
from onmt.decoders.transformer_multi_enc import TransformerMultiSrcDecoder

str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder,
           "transformer-rnn": CombinedTransformerRnnDecoder,
           "transformer-multi-sources-decoder": TransformerMultiSrcDecoder,
           }

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec", 'TransformerMultiSrcDecoder']
