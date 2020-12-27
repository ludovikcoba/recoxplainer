import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader


class TestDataReader(unittest.TestCase):

    def setUp(self) -> None:
        self.data = DataReader(**cfg.testdata)

    def test_import(self):

        self.assertEqual(self.data.num_user, 249)
        self.assertEqual(self.data.num_item, 551)
        self.assertEqual(self.data.dataset.shape[0], 1000)
        self.assertEqual(self.data.dataset.shape[1], 4)
