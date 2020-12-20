import numpy as np
import pandas as pd


class DataReader:

    def __init__(self, cfg):
        self.config = cfg
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = pd.read_csv(**self.config,
                                        engine='python')
        return self._dataset

    @property
    def path(self):
        return self.config['path']

    @property
    def header(self):
        return self.config.get('header', 'infer')

    @property
    def names(self):
        return self.config.get('names')

    @property
    def sep(self):
        return self.config.get("sep", ",")

    def make_consecutive_ids_in_dataset(self):
        dataset = self.dataset.rename({
                    "userId": "user_id",
                    "itemId": "item_id"
                }, axis=1)

        user_id = dataset[['user_id']].drop_duplicates().reindex()
        num_user = len(user_id)

        user_id['userId'] = np.arange(num_user)
        self._dataset = pd.merge(
            dataset, user_id,
            on=['user_id'], how='left')

        item_id = dataset[['item_id']].drop_duplicates()
        num_item = len(item_id)
        item_id['itemId'] = np.arange(num_item)

        self._dataset = pd.merge(
            self._dataset, item_id,
            on=['item_id'], how='left')

        self.origina_user_id = user_id
        self.origina_item_id = item_id

        self._dataset = self.dataset[
            ['userId', 'itemId', 'rating', 'timestamp']
        ]
        return self

    def binarize(self):
        """binarize into 0 or 1, imlicit feedback"""
        self._dataset['rating'][self._dataset['rating'] > 0] = 1.0
        self._dataset = self._dataset[self._dataset['rating'] > 0]

    @property
    def num_user(self):
        user_id = self.dataset[['userId']].drop_duplicates().reindex()
        return len(user_id)

    @property
    def num_item(self):
        item_id = self.dataset[['itemId']].drop_duplicates().reindex()
        return len(item_id)

    @property
    def userIds(self):
        return self.dataset[['userId']]

    @property
    def itemIds(self):
        return self.dataset[['itemId']]

    def dataset_info(self):
        """Print the number users and the items domain."""
        print('Range of userId is [{}, {}]'.format(self.userIds.min(),
                                                   self.userIds.max()))
        print('Range of itemId is [{}, {}]'.format(self.itemIds.min(),
                                                   self.itemIds.max()))
