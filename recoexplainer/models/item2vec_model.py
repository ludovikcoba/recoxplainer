import torch.nn as nn


class Item2Vec(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.embedding = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim)
        self.fc = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.num_items)

    def forward(self, input_data):
        embedding = self.embedding(input_data)
        return self.fc(embedding)

    def item_embedding(self):
        return self.embedding.weight.detach()
