import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from get_data import get_cifar10, get_fashionMNIST

device = "cpu"

class RecPCN(object):
    def __init__(self, dim):
        self.dim = dim

        # initialize parameters
        bound = 1. / np.sqrt(dim)
        self.Wr = torch.zeros((dim, dim)).fill_diagonal_(0).to(device)
        
    def learning(self, X):
        preds = torch.matmul(X, self.Wr.t())
        errs = X - preds
        grad_Wr = torch.matmul(errs.T, X).fill_diagonal_(0)

        return grad_Wr

    def inference(self, X_c):
        errs_X = X_c - torch.matmul(X_c, self.Wr.t())
        delta_X = -errs_X

        return delta_X


learning_iters = 100
learning_lr = 0.001
inference_iters = 100
inference_lr = 0.1
sample_size = 50
batch_size = 1
cover_size = 12
image_size = 28

pcn = RecPCN(dim=image_size**2)
X, X_c, update_mask = get_fashionMNIST('./data', sample_size=sample_size, 
                                  batch_size=batch_size, 
                                  seed=10, 
                                  cover_size=cover_size, 
                                  device=device)

mse = []
for i in range(learning_iters):
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        grad_Wr = pcn.learning(data)
        pcn.Wr += learning_lr * grad_Wr

    X_recon = X_c.clone()
    for j in range(inference_iters):
        delta_X = pcn.inference(X_recon)
        X_recon += inference_lr * (delta_X * update_mask)

    mse.append(torch.mean((X_recon - X)**2))

fig, ax = plt.subplots(5, 3, figsize=(5,8))
for i in range(5):
    ax[i, 0].imshow(X[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
    ax[i, 0].axis('off')
    ax[i, 1].imshow(X_c[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
    ax[i, 1].axis('off')
    ax[i, 2].imshow(X_recon[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
    ax[i, 2].axis('off')
plt.show()
