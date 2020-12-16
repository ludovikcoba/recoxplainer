from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader

import unittest


class TestDataReader(unittest.TestCase):

    def test_import(self):
        data = DataReader(cfg.testdata)

        self.assertEqual(data.num_user, 5)
        self.assertEqual(data.num_item, 32)
        self.assertEqual(data.dataset.shape[0], 76)
        self.assertEqual(data.dataset.shape[1], 4)
