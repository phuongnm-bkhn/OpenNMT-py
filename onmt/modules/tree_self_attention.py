import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.8, no_cuda=False):
        super(GroupAttention, self).__init__()
        self.d_model = 256.
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.no_cuda = no_cuda

    def forward(self, context, eos_mask, prior):
        batch_size, seq_len = context.size()[:2]

        context = self.norm(context)

        a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1)).to(context.device)
        b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).to(context.device)
        c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1)).to(context.device)
        tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0)).to(context.device)

        # mask = eos_mask & (a+c) | b
        mask = eos_mask & ((a+c) == 1)

        key = self.linear_key(context)
        query = self.linear_query(context)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model

        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-9)
        neibor_attn = prior + (1. - prior) * neibor_attn

        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-9)

        return g_attn, neibor_attn
