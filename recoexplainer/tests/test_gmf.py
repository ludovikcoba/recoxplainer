import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader
from recoexplainer.models.gmf_model import GMFModel


class GMFTest(unittest.TestCase):

    def setUp(self) -> None:
        self.gmf = GMFModel(cfg.model)
        self.data = DataReader(cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_train_gmf(self):
        self.assertTrue(self.gmf.fit(self.data.dataset))
