import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader
from recoexplainer.models.als_model import ALS
from recoexplainer.recommender import RankPredictionsRecommender


class RecommenderImplicitTest(unittest.TestCase):

    def setUp(self):
        self.als = ALS(**cfg.model.als)

        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

        self.als.fit(self.data.dataset)

    def test_train_recommend_als(self):
        recommender = RankPredictionsRecommender(self.data, self.als)
        recommender.recommend_all()
