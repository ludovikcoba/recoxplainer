from alpha_recsys.models.mf_implicit_model import MFImplicitModel
import implicit


class BPR(MFImplicitModel):
    """"""
    def __init__(self, config):
        super(BPR, self).__init__(config)
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.latent_dim,
            learning_rate=self.learning_rate,
            regularization=self.reg_term,
            iterations=self.num_epochs
        )
