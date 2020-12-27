from tqdm.auto import tqdm
import numpy as np
import pandas as pd


class ALSExplain:
    def __init__(self):
        pass

    def explain_all(self, model, recommendations, data, n=10):

        explanations = []
        dataset = data.dataset
        self.num_items = data.num_item

        self.users = dataset.groupby(by='userId')

        with tqdm(total=recommendations.shape[0], desc="Computing explanations: ") as pbar:
            sparse_mtr = model.rearrange_dataset(data.dataset)

            for _, row in recommendations.iterrows():
                explanations.append(self.expl_recommendation_user(model,
                                                                  int(row.userId),
                                                                  int(row.itemId)))
                pbar.update()

        recommendations['explanations'] = explanations
        return recommendations

    def get_user_items(self, user_id):
        return self.users.get_group(user_id).itemId.values

    def expl_recommendation_user(self, model, user_id: int, item_id: int):
        """
        Measuring the contribution of each item to the recommendation.
        :param user_id:
        :param rec: recommendations dataframe
        :param rank: position of the item to be explained
        :return: returns a dataframe with the contribution to the recommendation of each previously interacted with item.
        """

        current_interactions = np.zeros(self.num_items)
        current_interactions[self.get_user_items(user_id)] = 1

        c_u = np.diag(current_interactions)

        y_t = model.item_embedding().transpose()
        temp = np.matmul(y_t, c_u)
        temp = np.matmul(temp, model.item_embedding())
        temp = temp + np.diag([model.reg_term] * model.latent_dim)

        if len(self.get_user_items(user_id)) > 1:
            weight_mtr = np.linalg.inv(temp)
        else:
            weight_mtr = np.linalg.pinv(temp)

        temp = np.matmul(model.item_embedding(), weight_mtr)

        sim_to_rec_id = temp.dot(model.item_embedding()[item_id, :])

        sim_to_rec_id = sim_to_rec_id[self.get_user_items(user_id)]

        contribution = {"item": self.get_user_items(user_id), "contribution": sim_to_rec_id}
        contribution = pd.DataFrame(contribution)
        contribution = contribution.sort_values(by=["contribution"], ascending=False)
        return {"item": contribution.item, "contribution": contribution.contribution}
