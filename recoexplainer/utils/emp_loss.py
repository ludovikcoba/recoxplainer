import torch


class EMFLoss(torch.nn.Module):
    def __init__(self):
        super(EMFLoss, self).__init__()

    def forward(self, ratings_pred, ratings, u, v, reg_term, expl, expl_reg_term):

        mse = (ratings_pred.view(-1) - ratings) ** 2
        u_l2 = reg_term * torch.norm(u, 2, -1)
        v_l2 = reg_term * torch.norm(v, 2, -1)
        expl_constraint = expl_reg_term * torch.norm(u - v, 1, -1) * expl

        loss = mse + u_l2 + v_l2 + expl_constraint

        return loss.mean()
