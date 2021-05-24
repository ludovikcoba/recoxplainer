import unittest

from recoxplainer.config import cfg
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.data_reader.user_item_dict import UserItemDict


class UserItemMatrixTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()

    def test_user_item_matrix(self):
        user_dict = UserItemDict(self.data.dataset)

        x = self.data.dataset.userId[0]
        y = self.data.dataset.itemId[0]
        v = self.data.dataset.rating[0]
        self.assertEqual(user_dict[x][y], v)
