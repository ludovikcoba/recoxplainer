import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from recoexplainer.data_reader.user_item_rating_dataset import UserItemRatingDataset
from recoexplainer.utils.torch_utils import use_optimizer
from .py_torch_model import PyTorchModel


class GMFModel(PyTorchModel):

    def __init__(self,
                 learning_rate: int,
                 weight_decay: int,
                 latent_dim: int,
                 epochs: int,
                 num_negative: int,
                 batch_size: int,
                 cuda: bool,
                 optimizer_name: str,
                 device_id=None):

        super().__init__(
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda,
            optimizer_name=optimizer_name,
            device_id=device_id)

        self.negative_sample_size = num_negative
        self.weight_decay = weight_decay

        self.affine_output = torch.nn.Linear(
            in_features=self.latent_dim,
            out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.criterion = nn.BCELoss()

    def fit(self, dataset_metadata):

        self.optimizer = use_optimizer(network=self,
                                       weight_decay=self.weight_decay,
                                       learning_rate=self.learning_rate,
                                       optimizer=self.optimizer_name
                                       )
        dataset = dataset_metadata.dataset

        num_users = dataset_metadata.num_user
        num_items = dataset_metadata.num_item

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=self.latent_dim)

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=self.latent_dim)

        self.negatives = self._sample_negative(dataset)

        with tqdm(total=self.epochs) as progress:
            for epoch in range(self.epochs):
                train_loader = self.instance_a_train_loader(dataset,
                                                            self.negative_sample_size,
                                                            self.batch_size)
                loss = self.train_an_epoch(train_loader)
                progress.update(1)
                progress.set_postfix({"loss": loss})

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

    def train_an_epoch(self, train_loader):
        self.train()
        cnt = 0
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
            cnt += 1
        return total_loss / cnt

    def train_single_batch(self, users, items, ratings):
        if self.cuda is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.optimizer.zero_grad()
        ratings_pred = self(users, items)
        loss = self.criterion(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

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
        dot = self.affine_output(element_product)
        rating = self.logistic(dot)
        return rating

