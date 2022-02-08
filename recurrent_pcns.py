import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

device = "cpu"

def get_cifar10(datapath, sample_size, batch_size, seed, cover_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(datapath, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(datapath, train=False, transform=transform, download=True)
    
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X = []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
    X = torch.cat(X, dim=0).squeeze().to(device) # size, 32, 32

    # create the corrupted version
    size = X.shape
    mask = torch.ones_like(X).to(device)
    mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] += 1
    update_mask = update_mask.reshape(-1, 32**2)

    X_c = (X * mask).to(device) # size, 32, 32

    X = X.reshape(-1, 32**2) # size, 1024
    X_c = X_c.reshape(-1, 32**2) # size, 1024

    return X, X_c, update_mask


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
sample_size = 100
batch_size = 1
cover_size = 12

pcn = RecPCN(dim=1024)
X, X_c, update_mask = get_cifar10('./data', sample_size=sample_size, batch_size=batch_size, seed=10, cover_size=cover_size)

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
    ax[i, 0].imshow(X[i].cpu().detach().numpy().reshape(32, 32), cmap='gray')
    ax[i, 0].axis('off')
    ax[i, 1].imshow(X_c[i].cpu().detach().numpy().reshape(32, 32), cmap='gray')
    ax[i, 1].axis('off')
    ax[i, 2].imshow(X_recon[i].cpu().detach().numpy().reshape(32, 32), cmap='gray')
    ax[i, 2].axis('off')
plt.show()
