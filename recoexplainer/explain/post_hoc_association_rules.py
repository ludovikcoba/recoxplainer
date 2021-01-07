from tqdm.auto import tqdm

from .explainer import Explainer


class ARPostHocExplainer(Explainer):
    def __init__(self, model, recommendations, data):
        super(ARPostHocExplainer, self).__init__(model, recommendations, data)
        self.AR = None

    def get_user_items(self, user_id):
        return self.users.get_group(user_id).itemId.values

    def explain_recommendation_to_user(self, user_id: int, item_id: int):
        pass
