import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class RecPCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # initialize parameters
        bound = 1. / np.sqrt(dim)
        self.Wr = nn.Parameter(torch.zeros((dim, dim)))

    def forward(self, X):
        return torch.matmul(X, self.Wr.t())
        
    def learning(self, X):
        preds = self.forward(X)
        errs = X - preds
        grad_Wr = torch.matmul(errs.T, X).fill_diagonal_(0)

        self.Wr.grad = -grad_Wr

    def inference(self, X_c):
        errs_X = X_c - self.forward(X_c)
        delta_X = -errs_X

        return delta_X

class AutoEncoder(nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        # initialize parameters
        self.We = nn.Linear(dim, latent_dim, bias=False)
        self.Wd = nn.Linear(latent_dim, dim, bias=False)

        self.tanh = nn.Tanh()
        
    def forward(self, X):
        return self.tanh(self.Wd(self.tanh(self.We(X))))
    