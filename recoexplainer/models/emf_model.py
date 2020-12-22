from recoexplainer.models.py_torch_model import PyTorchModel
from recoexplainer.utils.torch_utils import use_optimizer
from recoexplainer.data_reader.user_item_rating_dataset import UserItemRatingDataset

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np


class EMFModel(PyTorchModel):

    def __init__(self, config):

        super().__init__(config)

        self.reg_term = config.reg_term
        self.exp_reg_term = config.exp_reg_term
        self.positive_threshold = config.positive_threshold

        self.affine_output = nn.Linear(
            in_features=self.latent_dim,
            out_features=1)


        self.criterion = self.constrained_loss()

    def fit(self, dataset_metadata):

        self.optimizer = use_optimizer(self.config,
                                       self)
        self.dataset_metadata = dataset_metadata
        self.dataset = dataset_metadata.dataset
        # FIXME
        num_users = self.dataset_metadata.num_user
        num_items = self.dataset_metadata.num_item

        self.embedding_user = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=self.latent_dim)

        self.embedding_item = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=self.latent_dim)

        self.compute_explainability()

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

    def constrained_loss(self, users, items, ratings):
        user_embeddings = self.embedding_user(users)
        item_embeddings = self.embedding_item(items)
        ratings_pred = self(users, items)
        loss = (ratings_pred - ratings) ** 2 \
               + self.reg_term * torch.norm(user_embeddings, 2, -1) \
               + self.reg_term * torch.norm(item_embeddings, 2, -1) \
               + self.exp_reg_term * torch.abs(user_embeddings - item_embeddings) * self.explainability(users, items)

        return loss.mean()

    def compute_explainability(self):
        ds = self.dataset.pivot(index='userId', columns='itemId', values='rating')
        ds = ds.fillna(0)
        ds = ds.to_numpy()
        ds = torch.from_numpy(ds)
        ds = ds / ds.norm(dim=1)[:, None]
        sim = torch.mm(ds, ds.transpose(0, 1))

        for i in range(self.dataset.userId.max() + 1):
            pass
        # TODO: steopped here
        filter_dataset_on_threshold = self.dataset[
            self.dataset['rating'] >= self.positive_threshold
        ]


    @staticmethod
    def instance_a_train_loader(self, dataset, batch_size):
        """instance train loader for one training epoch"""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(dataset.userId),
                                        item_tensor=torch.LongTensor(dataset.itemId),
                                        target_tensor=torch.FloatTensor(dataset.ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_single_batch(self, users, items, ratings):
        if self.cuda is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        self.optimizer.zero_grad()

        loss = self.criterion(users, users, ratings)

        loss.backward()

        self.optimizer.step()

        loss = loss.item()

        return loss

    def forward(self, user_indices, item_indices):
        user_embeddings = self.embedding_user(user_indices)
        item_embeddings = self.embedding_item(item_indices)
        rating = torch.affine_output(user_embeddings, item_embeddings)
        return rating
