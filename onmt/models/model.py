""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, **kwargs):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths, **kwargs.get("encoder", {}))

        if bptt is False:
            if "CombinedTransformerRnnEncoder" in str(type(self.encoder)):
                if "CombinedTransformerRnnDecoder" in str(type(self.decoder)):
                    for i, layer in enumerate(self.encoder.transformer):
                        enc_final_state = layer.encoder_state["final_state"]
                        enc_memory_bank = layer.encoder_state["memory_bank"]
                        layer.encoder_state = {}
                        self.decoder.transformer_layers[i].feed_rnn_decoder\
                            .init_state(src, memory_bank, enc_final_state)
                    self.decoder.init_state(src, enc_memory_bank, enc_state)
                elif "InputFeedRNNDecoder" in str(type(self.decoder)):
                    enc_final_states = []
                    for i, layer in enumerate(self.encoder.transformer):
                        enc_final_state = layer.encoder_state["final_state"]
                        memory_bank = layer.encoder_state["memory_bank"]
                        enc_final_state = torch.cat(enc_final_state, dim=0)
                        layer.encoder_state = {}
                        enc_final_states.append(enc_final_state)
                    enc_state = tuple(enc_final_states)
            if "TransformerMultiEncoder" not in str(type(self.encoder)):
                self.decoder.init_state(src, memory_bank, enc_state)

        if "TransformerMultiEncoder" in str(type(self.encoder)):
            if bptt is False:
                self.decoder.init_state(src, memory_bank[0], enc_state[0])
            src_dec_out, src_attns = self.decoder(dec_in, memory_bank[0],
                                                  memory_lengths=lengths[0],
                                                  with_align=with_align)

            soft_tgt_templ, lengths_soft_tgt_templ = kwargs.get("encoder", {}).get("soft_tgt_templ")
            if bptt is False:
                self.decoder.init_state(soft_tgt_templ, memory_bank[1], enc_state[1])
            soft_tgt_templ_dec_out, soft_tgt_templ_attns = self.decoder(dec_in, memory_bank[1],
                                                                        memory_lengths=lengths_soft_tgt_templ,
                                                                        with_align=with_align)

            dec_out = src_dec_out * 0.5 + soft_tgt_templ_dec_out * 0.5
            attns = src_attns
        else:
            dec_out, attns = self.decoder(dec_in, memory_bank,
                                          memory_lengths=lengths,
                                          with_align=with_align)
        return dec_out, attns, memory_bank

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)