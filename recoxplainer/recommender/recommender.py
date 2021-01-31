import pandas as pd

from .genericrecommender import GenericRecommender


class Recommender(GenericRecommender):

    def __init__(self, dataset_metadata, model, top_n: int = 10):
        super(Recommender, self).__init__(dataset_metadata, model, top_n)

    def get_predictions(self,
                        user_id: int,
                        target_item_id: list, ):
        predictions = self.model.predict(user_id, target_item_id)
        return predictions

    def recommend(self, user_id: int,
                  target_item_id: list):
        """
        Generate recommendations on specific itemId and userId
        :param user_id: list, user Ids
        :param target_item_id: list, item Ids
        :param rated_items: list, of rated interactions.
        :return: data.frame [userId, itemId, rank], recommendations ranking for the specified pairs of userId and itemId.
        """
        predictions = self.get_predictions(user_id, target_item_id)

        return self.rank_prediction(user_id, target_item_id, predictions)

    def recommend_user(self, user_id: int = None, user_ratings: pd.DataFrame = None):
        """
        Get recommendations for a user.
        :param user_id: int, a user Id
        :param user_ratings: list, interactions on the user
        :return: dataframe [userId, itemId, rank], recommendations ranking for the specified userId.
        """

        if user_ratings is None:
            user_ratings = self.get_rated(user_id=user_id)

        unrated_item_id = self.get_unrated(user_ratings['itemId'])

        return self.recommend(user_id=user_id,
                              target_item_id=unrated_item_id)
