import numpy as np
import pandas as pd


class ExplanationEvaluator:

    def __init__(self, num_users, top_n: int = 10):

        self._top_n = top_n
        self.num_users = num_users

    @property
    def top_n(self):
        return self._top_n

    @top_n.setter
    def top_n(self, top_n: int):
        self._top_n = top_n

    def mean_explaianable_precision(self, recommendations, explainability_matrix):

        recommendations['expl'] = explainability_matrix[
            [int(u) for u in recommendations.userId],
            [int(i) for i in recommendations.itemId]]
        recommendations = recommendations[recommendations.expl > 0]
        mep = recommendations.groupby('userId')['itemId'].count() / self.top_n
        mep = sum(mep)/self.num_users

        return mep

    def model_fidelity(self, explanations):
        explanations = explanations[[len(x) > 0 for x in explanations.explanations]]
        fidelity = explanations.groupby('userId')['itemId'].count() / self.top_n

        return sum(fidelity)/self.num_users

