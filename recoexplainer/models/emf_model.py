import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from recoexplainer.data_reader.user_item_rating_dataset import UserItemRatingDataset
from recoexplainer.utils.emp_loss import EMFLoss
from recoexplainer.utils.torch_utils import use_optimizer
from .py_torch_model import PyTorchModel


class EMFModel:

    def __init__(self,
                 learning_rate: float,
                 reg_term: float,
                 expl_reg_term: float,
                 positive_threshold: float,
                 latent_dim: int,
                 epochs: int,
                 knn: int
                 ):

        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.dataset = None
        self.dataset_metadata = None
        self.embedding_user = None
        self.embedding_item = None
        self.optimizer = None

        self.reg_term = reg_term
        self.expl_reg_term = expl_reg_term
        self.positive_threshold = positive_threshold
        self.knn = knn

        self.explainability_matrix = None
        self.sim_users = {}

        self.affine_output = nn.Linear(
            in_features=self.latent_dim,
            out_features=1)

        self.criterion = EMFLoss()

    def fit(self, dataset_metadata):

        self.dataset_metadata = dataset_metadata
        self.dataset = dataset_metadata.dataset

        num_users = self.dataset_metadata.num_user
        num_items = self.dataset_metadata.num_item

        self.embedding_user = np.random.uniform(low=0,
                                                high=0.5/self.latent_dim,
                                                size=(num_users, self.latent_dim))

        self.embedding_item = np.random.uniform(low=0,
                                                high=0.5/self.latent_dim,
                                                size=(num_items, self.latent_dim))

        self.compute_explainability()

        with tqdm(total=self.epochs) as progress:
            for epoch in range(self.epochs):
                self.dataset = self.dataset.sample(frac=1)
                loss = []
                for _, row in self.dataset.iterrows():
                    p_ui = self.predict(row.userId, row.itemId)

                    e_ui = row.rating - p_ui

                    loss.append(e_ui ** 2)


                    # u′i=ui+η·(2·(rij−ui·vTj)·vj−β·ui−λ·sgn(ui−vj)·Eij)
                    delta_u = 2 * e_ui * self.embedding_item[row.itemId, :]
                    delta_u -= self.reg_term * self.embedding_user[row.userId, :]
                    temp = np.sign(self.embedding_item[row.itemId, :] - self.embedding_user[row.userId, :])
                    temp *= self.expl_reg_term * self.explainability_matrix[row.userId, row.itemId]
                    delta_u -= temp

                    # v′j=vj+η·(2·(rij−ui·vTj)·ui−β·vj−λ·sgn(ui−vj)·Eij)
                    delta_v = 2 * e_ui * self.embedding_user[row.userId, :]
                    delta_v -= self.reg_term * self.embedding_item[row.itemId, :]
                    temp = np.sign(self.embedding_user[row.userId, :] - self.embedding_item[row.itemId, :])
                    temp *= self.expl_reg_term * self.explainability_matrix[row.userId, row.itemId]
                    delta_v -= temp

                    self.embedding_user[row.userId, :] += self.learning_rate * delta_u
                    self.embedding_item[row.itemId, :] += self.learning_rate * delta_v

                progress.update(1)

                progress.set_postfix({"MSE": sum(loss)/len(loss)})
        return True

    def compute_explainability(self):
        ds = self.dataset.pivot(index='userId', columns='itemId', values='rating')
        ds = ds.fillna(0)
        ds = sparse.csr_matrix(ds)
        sim_matrix = cosine_similarity(ds)
        min_val = sim_matrix.min() - 1

        for i in range(self.dataset_metadata.num_user):
            sim_matrix[i, i] = min_val

            knn_to_user_i = (-sim_matrix[i, :]).argsort()[:self.knn]
            self.sim_users[i] = knn_to_user_i

        self.explainability_matrix = np.zeros((self.dataset_metadata.num_user,
                                               self.dataset_metadata.num_item))

        filter_dataset_on_threshold = self.dataset[
            self.dataset['rating'] >= self.positive_threshold
            ]

        for i in range(self.dataset_metadata.num_user):
            knn_to_user_i = self.sim_users[i]

            rated_items_by_sim_users = filter_dataset_on_threshold[
                filter_dataset_on_threshold['userId'].isin(knn_to_user_i)]

            sim_scores = rated_items_by_sim_users.groupby(by='itemId')
            sim_scores = sim_scores['rating'].sum()
            sim_scores = sim_scores.reset_index()

            self.explainability_matrix[i, sim_scores.itemId] = sim_scores.rating.to_list()

        self.explainability_matrix = MinMaxScaler().fit_transform(self.explainability_matrix)

    def predict(self, user_id, item_id):
        if isinstance(user_id, list) or isinstance(item_id, list):
            if not isinstance(user_id, list):
                user_id = [user_id]
            if not isinstance(item_id, list):
                item_id = [item_id]

            for u in user_id:
                pred = [np.dot(self.embedding_user[u, :], self.embedding_item[i, :])
                        for i in item_id]
        else:
            pred = np.dot(self.embedding_user[user_id, :], self.embedding_item[item_id, :])
        return pred

    def user_embedding(self):
        return self.embedding_user

    def item_embedding(self):
        return self.embedding_item


class EMFTorchModel(PyTorchModel):

    def __init__(self,
                 learning_rate: float,
                 reg_term: float,
                 expl_reg_term: float,
                 positive_threshold: float,
                 momentum: float,
                 weight_decay: float,
                 latent_dim: int,
                 epochs: int,
                 batch_size: int,
                 knn: int,
                 cuda: bool,
                 optimizer_name: str,
                 device_id=None
                 ):

        super().__init__(
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda,
            optimizer_name=optimizer_name,
            device_id=device_id
        )

        self.reg_term = reg_term
        self.expl_reg_term = expl_reg_term
        self.positive_threshold = positive_threshold
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.knn = knn

        self.explainability_matrix = None
        self.sim_users = {}

        self.affine_output = nn.Linear(
            in_features=self.latent_dim,
            out_features=1)

        self.criterion = EMFLoss()

    def fit(self, dataset_metadata):

        self.dataset_metadata = dataset_metadata
        self.dataset = dataset_metadata.dataset

        num_users = self.dataset_metadata.num_user
        num_items = self.dataset_metadata.num_item

        self.embedding_user = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=self.latent_dim)

        self.embedding_item = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=self.latent_dim)

        self.compute_explainability()

        self.optimizer = use_optimizer(network=self,
                                       learning_rate=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay,
                                       optimizer=self.optimizer_name)

        with tqdm(total=self.epochs) as progress:
            for epoch in range(self.epochs):
                train_loader = self.instance_a_train_loader(self.batch_size)
                loss = self.train_an_epoch(train_loader)
                progress.update(1)
                progress.set_postfix({"loss": loss})
        return True

    def compute_explainability(self):
        ds = self.dataset.pivot(index='userId', columns='itemId', values='rating')
        ds = ds.fillna(0)
        ds = sparse.csr_matrix(ds)
        sim_matrix = cosine_similarity(ds)
        min_val = sim_matrix.min() - 1

        for i in range(self.dataset_metadata.num_user):
            sim_matrix[i, i] = min_val

            knn_to_user_i = (-sim_matrix[i, :]).argsort()[:self.knn]
            self.sim_users[i] = knn_to_user_i

        self.explainability_matrix = np.zeros((self.dataset_metadata.num_user,
                                               self.dataset_metadata.num_item))

        filter_dataset_on_threshold = self.dataset[
            self.dataset['rating'] >= self.positive_threshold
            ]

        for i in range(self.dataset_metadata.num_user):
            knn_to_user_i = self.sim_users[i]

            rated_items_by_sim_users = filter_dataset_on_threshold[
                filter_dataset_on_threshold['userId'].isin(knn_to_user_i)]

            sim_scores = rated_items_by_sim_users.groupby(by='itemId')
            sim_scores = sim_scores['rating'].sum()
            sim_scores = sim_scores.reset_index()

            self.explainability_matrix[i, sim_scores.itemId] = sim_scores.rating.to_list()

        self.explainability_matrix = MinMaxScaler().fit_transform(self.explainability_matrix)

        self.explainability_matrix = torch.from_numpy(self.explainability_matrix)

    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(self.dataset.userId),
                                        item_tensor=torch.LongTensor(self.dataset.itemId),
                                        target_tensor=torch.FloatTensor(self.dataset.rating))
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

        user_embeddings = self.embedding_user(users)
        item_embeddings = self.embedding_item(items)

        loss = self.criterion(ratings_pred=ratings_pred,
                              ratings=ratings,
                              u=user_embeddings,
                              v=item_embeddings,
                              reg_term=self.reg_term,
                              expl=self.explainability_matrix[users, items],
                              expl_reg_term=self.expl_reg_term)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()

        return loss

    def forward(self, user_indices, item_indices):
        user_embeddings = self.embedding_user(user_indices)
        item_embeddings = self.embedding_item(item_indices)
        element_product = torch.mul(user_embeddings, item_embeddings)
        rating = self.affine_output(element_product)
        return rating
