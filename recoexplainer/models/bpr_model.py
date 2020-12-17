from recoexplainer.models.mf_implicit_model import MFImplicitModel
import implicit


class BPR(MFImplicitModel):
    """"""
    def __init__(self,
                 latent_dim,
                 reg_term,
                 learning_rate,
                 epochs,
                 **kwargs):

        super(BPR, self).__init__(latent_dim=latent_dim,
                                  reg_term=reg_term,
                                  learning_rate=learning_rate,
                                  epochs=epochs)

        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.latent_dim,
            learning_rate=self.learning_rate,
            regularization=self.reg_term,
            iterations=self.epochs
        )
