from recoexplainer.models.mf_implicit_model import MFImplicitModel
import implicit


class ALS(MFImplicitModel):

    def __init__(self, config):
        super(ALS, self).__init__()
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.latent_dim,
            regularization=self.reg_term,
            iterations=self.epochs
        )
