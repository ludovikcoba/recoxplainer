from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .explainer import Explainer


class KNNPostHocExplainer(Explainer):

    def __init__(self,
                 model,
                 recommendations,
                 data,
                 knn=10):

        super(KNNPostHocExplainer, self).__init__(model, recommendations, data)

        self.AR = None
        self.knn = knn
        self.knn_items_dict = None

    def get_rules_for_getting(self, item_id):
        if self.knn_items_dict is None:

            self.knn_items_dict = {}
            self.compute_knn_items_for_all_items()

        return self.knn_items_dict[item_id]

    def compute_knn_items_for_all_items(self):

        ds = self.dataset.pivot(index='itemId',
                                columns='userId',
                                values='rating')

        ds = ds.fillna(0)
        ds = sparse.csr_matrix(ds)
        sim_matrix = cosine_similarity(ds)
        min_val = sim_matrix.min() - 1

        for i in range(self.num_items):
            sim_matrix[i, i] = min_val
            knn_to_item_i = (-sim_matrix[i, :]).argsort()[:self.knn]
            self.knn_items_dict[i] = knn_to_item_i

    def explain_recommendation_to_user(self, user_id: int, item_id: int):

        user_ratings = self.get_user_items(user_id)

        sim_items = self.get_rules_for_getting(item_id)

        explanations = set(sim_items) & set(user_ratings)

        return explanations



