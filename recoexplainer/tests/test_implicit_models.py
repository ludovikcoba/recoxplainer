import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader
from recoexplainer.models.als_model import ALS
from recoexplainer.models.bpr_model import BPR
from recoexplainer.recommender import Recommender


class ALSTest(unittest.TestCase):

    def setUp(self):
        self.als = ALS(**cfg.model.als)
        self.bpr = BPR(**cfg.model.bpr)
        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_train_als(self):
        self.assertTrue(self.als.fit(self.data))
        recommender = Recommender(self.data, self.als)
        recommender.recommend_all()

    def test_train_bpr(self):
        self.assertTrue(self.bpr.fit(self.data))
        recommender = Recommender(self.data, self.bpr)
        recommender.recommend_all()

