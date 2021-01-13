import unittest

from recoexplainer.config import cfg
from recoexplainer.data_reader import DataReader
from recoexplainer.models import ALS
from recoexplainer.explain import ALSExplainer
from recoexplainer.recommender import Recommender
from recoexplainer.evaluator import Evaluator, Splitter
from recoexplainer.explain import ARPostHocExplainer, KNNPostHocExplainer


class ALSTest(unittest.TestCase):

    def setUp(self):
        self.als = ALS(**cfg.model.als)

        self.data = DataReader(**cfg.ml100k)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_explain_als(self):
        sp = Splitter()
        train, test = sp.split_leave_n_out(self.data, n=1)
        self.assertTrue(self.als.fit(train))
        recommender = Recommender(self.data, self.als)
        recommendations = recommender.recommend_all()

        evaluator = Evaluator(test)
        evaluator.cal_hit_ratio(recommendations)

        #explainer = ALSExplainer(self.als, recommendations, self.data)
        #explainer.explain_recommendations()

        KNNexplainer = KNNPostHocExplainer(self.als, recommendations, train)
        KNNexpl = KNNexplainer.explain_recommendations()
