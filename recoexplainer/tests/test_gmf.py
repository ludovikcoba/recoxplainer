import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader
from recoexplainer.models.gmf_model import GMFModel
from recoexplainer.recommender import Recommender


class GMFTest(unittest.TestCase):

    def setUp(self) -> None:
        self.gmf = GMFModel(**cfg.model.gmf)
        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_train_gmf(self):
        self.assertTrue(self.gmf.fit(self.data))
        recommender = Recommender(self.data, self.gmf)
        recommender.recommend_all()
