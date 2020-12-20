import torch

from recoexplainer.utils.torch_utils import use_optimizer, use_cuda


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
        self.reg_term = config.reg_term
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self.num_negative = config.num_negative
        self.batch_size = config.batch_size
        self.cuda = config.cuda

        super().__init__()

    def train_an_epoch(self, train_loader, epoch_id):
        self.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            if batch_id % 200 == 0:
                print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss

    def train_single_batch(self, users, items, ratings):
        if self.cuda is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.optimizer.zero_grad()
        ratings_pred = self(users, items)
        loss = self.criterion(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def user_embedding(self):
        return self.state_dict()['embedding_user.weight'].cpu().numpy()

    def item_embedding(self):
        return self.state_dict()['embedding_item.weight'].cpu().numpy()
