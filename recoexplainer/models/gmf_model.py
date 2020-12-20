from recoexplainer.models.py_torch_model import PyTorchModel
from recoexplainer.utils.torch_utils import use_optimizer
from recoexplainer.data_reader.user_item_rating_dataset import UserItemRatingDataset

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import random


class GMFModel(PyTorchModel):

    def __init__(self, config):

        super().__init__(config)

        self.dataset = None
        self.embedding_user = None
        self.embedding_item = None

        self.affine_output = torch.nn.Linear(
            in_features=self.latent_dim,
            out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.criterion = nn.BCELoss()

    def fit(self, dataset):

        self.optimizer = use_optimizer(self.config,
                                       self)

        self.dataset = dataset
        # FIXME
        num_users = self.dataset.userId.max() + 1
        num_items = self.dataset.itemId.max() + 1

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=self.latent_dim)

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=self.latent_dim)

        self.negative_sample_size = self.num_negative
        self.negatives = self._sample_negative(self.dataset)

        print('Range of userId is [{}, {}]'.format(
            self.dataset.userId.min(),
            self.dataset.userId.max()))
        print('Range of itemId is [{}, {}]'.format(
            self.dataset.itemId.min(),
            self.dataset.itemId.max()))

        for epoch in range(self.epochs):
            print('Epoch {} starts !'.format(epoch))
            print('-' * 80)
            train_loader = self.instance_a_train_loader(self.dataset,
                                                        self.num_negative,
                                                        self.batch_size)
            self.train_an_epoch(train_loader, epoch_id=epoch)

        return True

    def instance_a_train_loader(self, dataset, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(dataset, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items']\
            .apply(lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings \
            .groupby('userId')['itemId'] \
            .apply(set) \
            .reset_index() \
            .rename(columns={'itemId': 'interacted_items'})
        self.item_catalogue = set(ratings.itemId)
        interact_status['negative_items'] = interact_status['interacted_items'] \
            .apply(lambda x: self.item_catalogue - x)
        return interact_status[['userId', 'negative_items']]

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

