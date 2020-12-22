import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader.data_reader import DataReader
from recoexplainer.models.emf_model import EMFModel


class GMFTest(unittest.TestCase):

    def setUp(self) -> None:
        self.emf = EMFModel(cfg.model)
        self.data = DataReader(cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_train_gmf(self):
        self.assertTrue(self.emf.fit(self.data.dataset))
