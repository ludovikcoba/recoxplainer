import torch


class GMFModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        print(self.num_users)
        print(self.latent_dim)
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim)
        self.affine_output = torch.nn.Linear(
            in_features=self.latent_dim,
            out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def user_embedding(self):
        return self.state_dict()['embedding_user.weight'].cpu().numpy()

    def item_embedding(self):
        return self.state_dict()['embedding_item.weight'].cpu().numpy()
