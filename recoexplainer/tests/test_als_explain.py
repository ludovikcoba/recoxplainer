import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader
from recoexplainer.models.als_model import ALS
from recoexplainer.explain.model_based_als_explain import ALSExplain
from recoexplainer.recommender import RankPredictionsRecommender


class ALSTest(unittest.TestCase):

    def setUp(self):
        self.als = ALS(**cfg.model.als)

        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_explain_als(self):
        self.assertTrue(self.als.fit(self.data.dataset))
        recommender = RankPredictionsRecommender(self.data, self.als)
        recommendations = recommender.recommend_all()
        explainer = ALSExplain()
        explainer.explain_all(self.als, recommendations, self.data)

