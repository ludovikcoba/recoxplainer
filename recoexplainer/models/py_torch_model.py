import torch
import itertools
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

    def predict(self, user_id, item_id):
        if type(user_id) == 'int':
            user_id = [user_id]
        if type(item_id) == 'int':
            item_id = [item_id]
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor(item_id)
        with torch.no_grad():
            if self.cuda:
                user_id = user_id.cuda()
                item_id = item_id.cuda()
            pred = self.forward(user_id, item_id).cpu().tolist()
            pred = list(itertools.chain.from_iterable(pred))
            return pred

    def user_embedding(self):
        return self.state_dict()['embedding_user.weight'].cpu().numpy()

    def item_embedding(self):
        return self.state_dict()['embedding_item.weight'].cpu().numpy()
