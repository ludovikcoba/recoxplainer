from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

from .explainer import Explainer


class ARPostHocExplainer(Explainer):
    def __init__(self,
                 model,
                 recommendations,
                 data,
                 min_support=.1,
                 max_len=2,
                 metric="lift",
                 min_threshold=.1):

        super(ARPostHocExplainer, self).__init__(model, recommendations, data)
        self.AR = None
        self.min_support = min_support
        self.max_len = max_len
        self.metric = metric
        self.min_threshold = min_threshold

        self.rules = None

    def get_rules_for_getting(self, item_id):
        if self.rules is None:
            self.compute_association_rules()

        return self.rules[self.rules.consequents == item_id]

    def compute_association_rules(self):

        item_sets = [
            [item for item in self.dataset[self.dataset.userId == user].itemId]
            for user in self.dataset.userId.unique()
        ]

        te = TransactionEncoder()
        te_ary = te.fit(item_sets).transform(item_sets)

        df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(df,
                                    min_support=self.min_support,
                                    use_colnames=True,
                                    max_len=self.max_len)

        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=.1)
        rules = rules[(rules['confidence'] > 0.1) &
                      (rules['lift'] > 0.1)]

        rules.consequents = [list(row.consequents)[0] for _, row in rules.iterrows()]
        rules.antecedents = [list(row.antecedents)[0] for _, row in rules.iterrows()]

        self.rules = rules[["consequents", "antecedents", "confidence"]]

    def explain_recommendation_to_user(self, user_id: int, item_id: int):

        user_ratings = self.get_user_items(user_id)

        rules = self.get_rules_for_getting(item_id)

        explanations = rules[rules.antecedents.isin(user_ratings)]

        return {"item": explanations.antecedents,
                "confidence": explanations.confidence}



