import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader
from recoexplainer.models.als_model import ALS
from recoexplainer.models.bpr_model import BPR


class ALSTest(unittest.TestCase):

    def setUp(self):
        self.als = ALS(**cfg.model)
        self.bpr = BPR(**cfg.model)
        self.data = DataReader(cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_train_als(self):
        self.assertTrue(self.als.fit(self.data.dataset))

    def test_train_bpr(self):
        self.assertTrue(self.bpr.fit(self.data.dataset))
