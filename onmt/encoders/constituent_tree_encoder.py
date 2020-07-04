from torch import nn


class ConstituentTreeEncoder(nn.Module):
    def __init__(self, tree_field, embedding_dim, hidden_size, num_layer=1):
        super(ConstituentTreeEncoder, self).__init__()
        self.emb = nn.Embedding(len(tree_field.vocab.stoi), embedding_dim=embedding_dim,
                                padding_idx=tree_field.vocab.stoi[tree_field.pad_token])

        self.input_size = embedding_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           batch_first=tree_field.batch_first,
                           num_layers=self.num_layer,
                           bidirectional=False)

    def forward(self, input_ids):
        out = self.emb(input_ids)
        out, (h_n, c_n) = self.rnn(out)
        return (h_n + c_n).squeeze(dim=0)
