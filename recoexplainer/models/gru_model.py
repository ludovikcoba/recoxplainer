import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        dict_size = config.get("dict_size")
        output_size = config.get("output_size")
        embedding_dim = config.get("embedding_dim")
        hidden_dim = config.get("hidden_dim")
        p_dropout = config.get("p_dropout", 0.5)
        self.embedding = nn.Embedding(dict_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        embeded = self.embedding(x)
        rnn_out, _ = self.rnn(embeded)
        rnn_out = rnn_out[:, -1]
        rnn_dropout = self.dropout(rnn_out)
        logits = self.fc(rnn_dropout)
        return logits
