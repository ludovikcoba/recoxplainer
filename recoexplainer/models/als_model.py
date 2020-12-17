from recoexplainer.models.mf_implicit_model import MFImplicitModel
import implicit


class ALS(MFImplicitModel):

    def __init__(self,
                 latent_dim,
                 reg_term,
                 epochs,
                 **kwargs):

        super(ALS, self).__init__(latent_dim=latent_dim,
                                  reg_term=reg_term,
                                  epochs=epochs,
                                  learning_rate=None)

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.latent_dim,
            regularization=self.reg_term,
            iterations=self.epochs
        )
