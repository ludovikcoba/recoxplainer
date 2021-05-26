from torch.utils.data import Dataset
import torch
import numpy as np


class UserItemDict(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, data, expl_matrix, expl):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """

        grp_data = data.groupby('userId')
        self.users_dict = dict()
        for userId, itemId_rating in grp_data:
            self.users_dict[userId] = {'items': list(itemId_rating.itemId),
                                       'rating': list(itemId_rating.rating)}
        self.n_items = data.itemId.nunique()
        self.n_users = data.userId.nunique()
        self.expl_matrix = expl_matrix
        self.expl = expl

    def __getitem__(self, index):
        ratings = np.zeros(self.n_items)
        ratings[self.users_dict[index]['items']] = self.users_dict[index]['rating']
        if self.expl:
            return torch.tensor(ratings) + self.expl_matrix[index, :]
        else:
            return torch.tensor(ratings)

    def __len__(self):
        return self.n_users

