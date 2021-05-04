import unittest

from recoxplainer.config import cfg
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.models.mlp_model import MLPModel
from recoxplainer.recommender import Recommender


class MLPTest(unittest.TestCase):

    def setUp(self) -> None:
        self.mlp = MLPModel(**cfg.model.mlp)
        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_train_mlp(self):
        self.assertTrue(self.mlp.fit(self.data))
        recommender = Recommender(self.data, self.mlp)
        recommender.recommend_all()
