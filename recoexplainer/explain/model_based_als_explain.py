import numpy as np
import pandas as pd

from .explainer import Explainer


class ALSExplainer(Explainer):
    def __init__(self, model, recommendations, data, number_of_contributions=10):
        super(ALSExplainer, self).__init__(model, recommendations, data)
        self.number_of_contributions = number_of_contributions

    def explain_recommendation_to_user(self, user_id: int, item_id: int):
        """
        Measuring the contribution of each item to the recommendation.
        :param model:
        :param item_id:
        :param user_id:
        :return: returns a dataframe with the contribution to the recommendation of each previously interacted with item.
        """

        current_interactions = np.zeros(self.num_items)
        current_interactions[self.get_user_items(user_id)] = 1

        c_u = np.diag(current_interactions)

        y_t = self.model.item_embedding().transpose()
        temp = np.matmul(y_t, c_u)
        temp = np.matmul(temp, self.model.item_embedding())
        temp = temp + np.diag([self.model.reg_term] * self.model.latent_dim)

        if len(self.get_user_items(user_id)) > 1:
            weight_mtr = np.linalg.inv(temp)
        else:
            weight_mtr = np.linalg.pinv(temp)

        temp = np.matmul(self.model.item_embedding(), weight_mtr)

        sim_to_rec_id = temp.dot(self.model.item_embedding()[item_id, :])

        sim_to_rec_id = sim_to_rec_id[self.get_user_items(user_id)]

        contribution = {"item": self.get_user_items(user_id), "contribution": sim_to_rec_id}
        contribution = pd.DataFrame(contribution)
        contribution = contribution.sort_values(by=["contribution"], ascending=False)
        return {"item": contribution.item[:self.number_of_contributions],
                "contribution": contribution.contribution[:self.number_of_contributions]}
