import numpy as np


class MFImplicitModel:
    def __init__(self,
                 latent_dim,
                 reg_term,
                 learning_rate,
                 epochs):

        self.latent_dim = latent_dim
        self.reg_term = reg_term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, dataset):
        self.model.fit(dataset)

    def predict(self, user_id, item_id):
        dot_prod = self.model.user_factors[user_id] * \
                   self.model.item_factors[item_id]
        return np.sum(dot_prod, axis=1)

    def user_embedding(self):
        return self.model.user_factors

    def item_embedding(self):
        return self.model.item_factors
