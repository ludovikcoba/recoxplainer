from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader

import unittest


class TestDataReader(unittest.TestCase):

    def setUp(self) -> None:
        self.data = DataReader(cfg.testdata)

    def test_import(self):
        data = DataReader(cfg.testdata)

        self.assertEqual(self.data.num_user, 250)
        self.assertEqual(self.data.num_item, 552)
        self.assertEqual(self.data.dataset.shape[0], 1001)
        self.assertEqual(self.data.dataset.shape[1], 4)
