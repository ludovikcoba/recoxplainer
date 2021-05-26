import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from recoxplainer.utils.torch_utils import use_cuda
from recoxplainer.data_reader.user_item_dict import UserItemDict
from recoxplainer.utils.torch_utils import use_optimizer


class ExplAutoencoderTorch(nn.Module):
    def __init__(self,
                 hidden_layer_features: int,
                 learning_rate: float,
                 positive_threshold: float,
                 weight_decay: float,
                 epochs: int,
                 knn: int,
                 cuda: bool,
                 optimizer_name: str,
                 expl: bool,
                 device_id=None
                 ):
        if optimizer_name not in ['sgd', 'adam', 'rmsprop']:
            raise Exception["Wrong optimizer."]
        if cuda is True:
            use_cuda(True, device_id)

        self.positive_threshold = positive_threshold
        self.weight_decay = weight_decay
        self.knn = knn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cuda = cuda
        self.optimizer_name = optimizer_name
        self.hidden_layer_features = hidden_layer_features
        self.expl = expl

        self.dataset = None
        self.dataset_metadata = None
        self.embedding_user = None
        self.embedding_item = None
        self.optimizer = None

        self.explainability_matrix = None
        self.sim_users = {}

        super().__init__()
        self.criterion = nn.MSELoss()

    def fit(self, dataset_metadata):
        self.dataset_metadata = dataset_metadata
        self.dataset = dataset_metadata.dataset

        num_items = self.dataset_metadata.num_item

        self.encoder_hidden_layer = nn.Linear(
            in_features=num_items, out_features=self.hidden_layer_features
        )

        self.decoder_output_layer = nn.Linear(
            in_features=self.hidden_layer_features, out_features=num_items
        )

        self.compute_explainability()

        self.optimizer = use_optimizer(network=self,
                                       learning_rate=self.learning_rate,
                                       weight_decay=self.weight_decay,
                                       optimizer=self.optimizer_name)

        with tqdm(total=self.epochs) as progress:
            train_loader = self.instance_a_train_loader()
            for epoch in range(self.epochs):
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

    def instance_a_train_loader(self):
        """instance train loader for one training epoch"""
        self.user_item_dict = UserItemDict(self.dataset, self.explainability_matrix, self.expl)
        return DataLoader(self.user_item_dict, shuffle=True)

    def train_an_epoch(self, train_loader):
        self.train()
        cnt = 0
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.Tensor)
            rating = batch[0]
            rating = rating.float()
            loss = self.train_single_user(rating)
            total_loss += loss
            cnt += 1
        return total_loss / cnt

    def train_single_user(self, ratings):
        if self.cuda is True:
            ratings = ratings.cuda()

        self.optimizer.zero_grad()
        ratings_pred = self(ratings)
        loss = self.criterion(ratings_pred, ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def forward(self, user_adjusted_ratings):
        activation = self.encoder_hidden_layer(user_adjusted_ratings)
        code = torch.relu(activation)
        activation = self.decoder_output_layer(code)
        reconstructed_ratings = torch.relu(activation)
        return reconstructed_ratings

    def predict(self, user_id, item_id):
        if type(user_id) == 'int':
            user_id = [user_id]
        if type(item_id) == 'int':
            item_id = [item_id]
        with torch.no_grad():
            if self.cuda:
                user_id = user_id.cuda()
                item_id = item_id.cuda()
            rating = self.user_item_dict[user_id]
            rating = rating.float()
            pred = self.forward(rating).cpu()
            return pred[item_id].tolist()
