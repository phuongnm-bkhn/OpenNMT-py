import random

import torch
import torchtext
from torch import nn

from onmt.inputters.structure_text_dataset import TreeField

random.seed(100)
torch.manual_seed(100)


class ConstituentTreeEncoder(nn.Module):
    def __init__(self, tree_field, embedding_dim, hidden_size, batch_first=False, num_layer=1):
        super(ConstituentTreeEncoder, self).__init__()
        self.emb = nn.Embedding(len(tree_field.vocab.stoi), embedding_dim=embedding_dim,
                                padding_idx=tree_field.vocab.stoi[tree_field.pad_token])

        self.input_size = embedding_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           batch_first=batch_first,
                           num_layers=self.num_layer,
                           bidirectional=False)

    def forward(self, input_ids):
        out = self.emb(input_ids)
        out, (h_n, c_n) = self.rnn(out)
        return (h_n + c_n).squeeze(dim=0)


embedding_dim = 9
input_ids = torch.LongTensor([[4, 2, 1, 1], [2, 2, 3, 1]])
print(input_ids)

base_feat = TreeField(
    sep="<sep>",
    # init_token=bos, eos_token=eos,
    pad_token="<pad>", tokenize=None,
    include_lengths=True,
    lower=True
)

vocab = torch.load(
    "/Volumes/GoogleDrive/My Drive/MacbookPro/SourcesCode/Master/data-sem/atis_unstemmed/minidata.vocab.pt")
train_dat = torch.load(
    "/Volumes/GoogleDrive/My Drive/MacbookPro/SourcesCode/Master/data-sem/atis_unstemmed/minidata.train.1.pt")
m = ConstituentTreeEncoder(vocab["constituent_tree"].base_field,
                           embedding_dim=embedding_dim,
                           batch_first=False,
                           hidden_size=11)

# out = m(input_ids)
train_dat.fields= vocab
mini_b = torchtext.data.Batch(
    train_dat.examples[99:128],
    train_dat,)
out = m(mini_b.constituent_tree.squeeze(dim=2))
print(out)
print(out.shape)
