import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, in_dim, out_dim=256, p=0.1, features=2, device=None, stds=None, alpha=1e-2, beta=1e-2, W=None):
        super(Model, self).__init__()
        self.device = device
        self.features = features.to(self.device)
        # self.out_dim = out_dim
        self.p = p
        self.dp = nn.Dropout(p=self.p)
        self.mse_loss = nn.MSELoss()

        """ a representer and predictors """
        self.representer = MLP(in_dim, out_dim, p=self.p)
        self.treat_mlp = MLP(out_dim, 1, p=self.p)
        self.cont_mlp = MLP(out_dim, 1, p=self.p)

        """ regularizers """
        self.alpha = alpha
        self.beta = beta
        stds = torch.Tensor(stds)
        self.treat_std, self.cont_std = stds[0].to(self.device), stds[1].to(self.device)

        """ a similarity matrix """
        self.W = W.to(self.device)

        self = self.to(self.device)

    def forward(self, x, node_idx, t, y):
        x = self.representer(x)
        x = self.dp(x)
        x = F.relu(x)
        x_treat2 = self.treat_mlp(x)
        x_cont2 = self.cont_mlp(x)
        predicted = torch.cat((x_treat2[t==1], x_cont2[t==0]), dim=0)
        mse_loss = self.mse_loss(predicted.view(-1), torch.cat((y[t==1], y[t==0]), dim=0))
        return mse_loss

    def outcome_smoothness_loss(self, node_idx, neighbors):
        x_treat, x_cont = self.predict(node_idx)
        weight = self.W[node_idx, neighbors.view(-1)]
        neighbors_treat, neighbors_cont = self.predict(neighbors)
        outcome_smoothness_treat = self.outcome_smoothness(x_treat, neighbors_treat, weight)
        outcome_smoothness_cont = self.outcome_smoothness(x_cont, neighbors_cont, weight)
        return (1/(self.treat_std**2))*outcome_smoothness_treat + (1/(self.cont_std**2))*outcome_smoothness_cont

    def treatment_effect_smoothness_loss(self, node_idx, neighbors):
        x_treat, x_cont = self.predict(node_idx)
        weight = self.W[node_idx, neighbors.view(-1)]
        neighbors_treat, neighbors_cont = self.predict(neighbors)
        treatment_effect_smoothness = (1/(self.treat_std**2+self.cont_std**2))*self.treatment_effect_smoothness(x_treat, x_cont, neighbors_treat, neighbors_cont, weight)
        return treatment_effect_smoothness

    def outcome_smoothness(self, x, neighbors, weight):
        weight = weight.view(-1)
        return self.alpha*torch.mean(weight*((x-neighbors)**2))

    def treatment_effect_smoothness(self, x_treat, x_cont, neighbors_treat, neighbors_cont, weight):
        weight = weight.view(-1)
        return self.beta*torch.mean(weight*(((neighbors_treat.view(-1)-neighbors_cont.view(-1))-(x_treat.view(-1)-x_cont.view(-1)))**2))

    def predict(self, node_idx):
        x = self.features[node_idx]
        x = self.representer(x)
        x = self.dp(x)
        x = F.relu(x)
        x_treat = self.treat_mlp(x)
        x_cont = self.cont_mlp(x)
        return x_treat.view(-1), x_cont.view(-1)

"""
    MLP representer
"""
class MLP (nn.Module):
    def __init__(self, in_dim=32, out_dim=32, p=0., norm=False):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        self.l2 = nn.Linear(in_dim, out_dim)
        self.dp = nn.Dropout(p=p)
        torch.nn.init.normal_(self.l1.weight, mean=0.0, std=0.1/np.sqrt(in_dim))
        torch.nn.init.normal_(self.l2.weight, mean=0.0, std=0.1/np.sqrt((in_dim)))

    def forward(self, x):
        x = self.l1(x)
        x = self.dp(x)
        x = F.relu(x)
        return self.l2(x)
