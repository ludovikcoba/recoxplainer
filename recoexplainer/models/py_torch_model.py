import torch

from recoexplainer.utils.torch_utils import use_cuda


class PyTorchModel(torch.nn.Module):
    """Meta Learner

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):

        if config.optimizer not in ['sgd', 'adam', 'rmsprop']:
            raise Exception["Wrong optimizer."]

        if config.cuda is True:
            use_cuda(True, config.device_id)
            self.cuda()

        self.config = config

        self.latent_dim = config.latent_dim
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.cuda = config.cuda

        self.dataset = None
        self.dataset_metadata = None
        self.embedding_user = None
        self.embedding_item = None

        super().__init__()

    def user_embedding(self):
        return self.state_dict()['embedding_user.weight'].cpu().numpy()

    def item_embedding(self):
        return self.state_dict()['embedding_item.weight'].cpu().numpy()
