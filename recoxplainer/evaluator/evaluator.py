import numpy as np
import pandas as pd


class Evaluator:
    disc_functions = ['log', 'linear']

    def __init__(self, test_set, top_n: int = 10, discount_function: str = 'log'):
        self.test_set = test_set
        self._top_n = top_n
        assert discount_function in self.disc_functions, "Wrong Discount Function."
        self._discount_function = discount_function
        self.num_users = self.test_set.userId.nunique()

    @property
    def top_n(self):
        return self._top_n

    @top_n.setter
    def top_n(self, top_n: int):
        self._top_n = top_n

    @property
    def discount_function(self):
        return self._discount_function

    @discount_function.setter
    def discount_function(self, discount_function: str):
        assert discount_function in self.disc_functions, "Wrong Discount Function."
        self._discount_function = discount_function

    def cal_hit_ratio(self, recommendations):
        """
        Hit Ratio
        :param recommendations: dataframe, columns = ['userId', 'itemId', 'rank']
        :return: hit rate.
        """
        test_in_top_n = self.get_hits(recommendations)
        # count hits per user
        hits_per_user = self.count_positives(test_in_top_n)
        # merge with the entire list of positive items for user
        hits_per_user = hits_per_user.merge(self.count_positives(self.test_set),
                                            on='userId',
                                            suffixes=('_true', ''),
                                            how='right')
        # if there are users with 0 hits the merge will have NA.
        hits_per_user = hits_per_user.fillna(0)
        # get the hit rate per user
        hit_rate = hits_per_user.positive_true / hits_per_user.positive
        # average
        hit_rate = hit_rate.mean()
        return hit_rate

    def get_hits(self, recommendations):
        """
        Find which items in the test set have a hit on the recommendations.
        :param recommendations: dataframe, columns = ['userId', 'itemId', 'rank']
        :return: dataframe, removing the rows missing in the test set.
        """
        # check whether there are top_n items per user
        top_n_recommendations = self.filter_to_top_n(recommendations)
        # find the hits
        test_in_top_n = pd.merge(top_n_recommendations, self.test_set,
                                 on=['userId', 'itemId'])
        return test_in_top_n

    def filter_to_top_n(self, dataset):
        """
        if rank > top_n, we do not use it for evaluation
        :param dataset: dataframe, columns = ['userId', 'itemId', 'rank']
        :return: dataframe, columns = ['userId', 'itemId', 'rank']
        """
        return dataset[dataset['rank'] <= self.top_n]

    def cal_ndcg(self, recommendations):
        """
        For evaluating the top-N recommendation list, we also provide the normalized Discounted Cumulative Gain at N
        recommendation (nDCG@N)  computed as the ratio of the Discounted Cumulative Gain(DCG) with the ideal Discounted
        Cumulative Gain(IDCG):
         DGC_{pos} = rel_1 + \sum_{i=2}^{pos} \frac{rel_i}{\log_2i} \qquad \qquad
        IDGC_{pos} = rel_1 + \sum_{i=2}^{|h|-1} \frac{rel_i}{\log_2i} \\
        nDCG_{pos} = \frac{DCG}{IDCG}
        where pos denotes the position up to which relevance is accumulated, and $rel_i$ is the relevance of the recommended item at position \textit{i}.
        Ref: Y. Wang, L. Wang, Y. Li, D. He, T.-Y. Liu, and W. Chen.
            A theoretical analysis of ndcgtype ranking measures.
        :param recommendations: dataframe, columns = ['userId', 'itemId', 'rank']
        :return: nDCG
        """
        # get hits
        hits = self.get_hits(recommendations)

        DCG = self.cal_dcg(hits)
        iDCG = self.cal_idcg()

        # join to check if there are users in the test without hits
        nDCG = iDCG.merge(DCG, on='userId', how='left')
        nDCG = nDCG.fillna(0)
        # normalize
        nDCG['ndcg'] = nDCG['dcg'] / nDCG['idcg']

        return nDCG['ndcg'].mean()

    def cal_dcg(self, hits):
        """
        Discounted Comulative Gain
        :param hits: recommendations: dataframe, columns = ['userId', 'itemId', 'rank']
        :return: DCG
        """
        # todo: the gain so far is set to a constant.

        if self.discount_function == 'log':
            hits['discounted_gain'] = np.log(2) / np.log(hits['rank'] + 1)
        elif self.discount_function == 'linear':
            hits['discounted_gain'] = 1 / hits['rank']

        DCG = hits.groupby('userId')['discounted_gain'].sum()

        return pd.DataFrame({'userId': hits['userId'].unique(), 'dcg': DCG}).reset_index(drop=True)

    def cal_idcg(self):
        """
        the Ideal DCG, is the DCG for the best ranking possible (i.e. all true positives were recommended first).
        :return: iDCG
        """
        # create a fake ranking for test set items.
        # We assume that the items in the test set are all on the Top-N list.
        count_positives = self.count_positives(self.test_set)
        ideal_rank = [i for x in count_positives['positive'] for i in (range(1, x + 1))]
        test_ideal_ranking = self.test_set.copy()
        test_ideal_ranking['rank'] = ideal_rank
        # Filter to have at most top-N items.
        test_ideal_ranking = self.filter_to_top_n(test_ideal_ranking)
        # get the dcg for the ideal ranking
        idcg = self.cal_dcg(test_ideal_ranking)
        idcg = idcg.rename(columns={'dcg': 'idcg'})
        return idcg

    @staticmethod
    def count_positives(dataset):
        """
        Returns the positives count.
        :param dataset: dataframe, columns = ['userId', 'itemId', 'rank']
        :return: dataframe, columns = ['userId', 'positive']
        """
        users_with_positives = dataset.userId.unique()
        positives_per_user = dataset.groupby('userId')['itemId'].count()
        positives_per_user = pd.DataFrame({'userId': users_with_positives,
                                           'positive': positives_per_user})

        return positives_per_user.reset_index(drop=True)


#if __name__ == '__main__':
##    recoms = pd.DataFrame({
#        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
#        'itemId': [1, 2, 3, 4, 1, 2, 2, 3, 4],
#        'rank': [1, 2, 3, 1, 2, 3, 1, 2, 3]
#    })

#    test = pd.DataFrame({
#        'userId': [1, 1, 2, 3],
#        'itemId': [1, 4, 1, 5]
#    })

#    eval = Evaluator(test_set=test, top_n=2)

#    assert eval.num_users == 3, 'number of users'
#    assert eval.top_n == 2, 'number of top n'
#    eval.top_n = 3
#    assert eval.top_n == 3, 'changing of top n'

 #   print(eval.cal_hit_ratio(recoms))
 #   print(eval.cal_ndcg(recoms))
