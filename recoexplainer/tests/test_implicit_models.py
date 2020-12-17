import unittest
from recoexplainer.config import cfg
from recoexplainer.models.als_model import ALS
from recoexplainer.models.bpr_model import BPR


class ALSTest(unittest.TestCase):

    def SetUp(self):
        self.als = ALS(**cfg.model)
        self.bpr = BPR(**cfg.model)

    def TestTrainALS(self):
        self.assertTrue(self.als.fit())

    def TestTrainBPR(self):
        self.assertTrue(self.bpr.fit())

