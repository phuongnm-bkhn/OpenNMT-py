"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class TransformerMultiEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0, gram_sizes=None):
        super(TransformerMultiEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions, gram_sizes=gram_sizes)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.save_self_attn = False
        self.self_attn_data = None

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, self_attn_data = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        if self.save_self_attn:
            self.self_attn_data = self_attn_data

        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

    def clean_self_attn_data(self):
        self.self_attn_data = None


class TransformerMultiEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 gram_sizes=None,
                 # src_constituent_tree_emb=None,
                 soft_tgt_templ_emb=None,
                 ):
        super(TransformerMultiEncoder, self).__init__()

        self.embeddings = embeddings
        # self.constituent_tree_emb = src_constituent_tree_emb
        self.transformer = nn.ModuleList(
            [TransformerMultiEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions, gram_sizes=gram_sizes)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.soft_tgt_templ_emb = soft_tgt_templ_emb
        self.transformer_soft_tgt_templ = nn.ModuleList(
            [TransformerMultiEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions, gram_sizes=gram_sizes)
                for i in range(num_layers)])
        self.layer_norm_soft_tgt_templ = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings, *args, **kwargs):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            gram_sizes=opt.gram_sizes,
            *args, **kwargs
        )

    def forward(self, src, lengths=None, **kwargs):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        # constituent_tree = kwargs.get("constituent_tree")
        # constituent_tree_feat = self.constituent_tree_emb(constituent_tree.squeeze(dim=2))
        soft_tgt_templ, lengths_soft_tgt_templ = kwargs.get("soft_tgt_templ")

        emb = self.embeddings(src)
        soft_tgt_templ_emb = self.soft_tgt_templ_emb(soft_tgt_templ)
        # emb = emb + constituent_tree_feat.reshape(emb.shape)

        def transformer_identical_forward(emb, transformer_blocks, layer_norm, lengths):
            out = emb.transpose(0, 1).contiguous()
            mask = ~sequence_mask(lengths).unsqueeze(1)
            # Run the forward pass of every layer of the tranformer.
            for layer in transformer_blocks:
                out = layer(out, mask)
            out = layer_norm(out)
            return out

        out_src = transformer_identical_forward(emb, self.transformer, self.layer_norm, lengths=lengths)
        out_soft_tgt_templ = transformer_identical_forward(soft_tgt_templ_emb, self.transformer_soft_tgt_templ,
                                                           self.layer_norm_soft_tgt_templ,
                                                           lengths=lengths_soft_tgt_templ)

        return (emb, soft_tgt_templ_emb), \
               (out_src.transpose(0, 1).contiguous(), out_soft_tgt_templ.transpose(0, 1).contiguous()), \
               (lengths, lengths_soft_tgt_templ)

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
