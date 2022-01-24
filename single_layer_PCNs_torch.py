from numpy.core.fromnumeric import mean
import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt


device = "cpu"
cover_size = 12
sample_size = 1000
batch_size = 1
NODES = 2

def get_cifar10(datapath, batch_size, seed, size=60000):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(datapath, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(datapath, train=False, transform=transform, download=True)
    
    if size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), size))
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

def get_gaussian_mixture(size):
    mean1 = np.array([3, 3])
    cov1 = np.array([[2, 0.5],
                    [0.5, 2]])
    mean2 = np.array([-3, -3])
    cov2 = np.array([[1, 0.2],
                    [0.2, 1]])
    X = np.empty((size, 2))
    np.random.seed(0)
    for i in range(size):
        rand_int = np.random.uniform()
        if rand_int > 0.5:
            X[i] = np.random.multivariate_normal(mean1, cov1, size=1)
        else:
            X[i] = np.random.multivariate_normal(mean2, cov2, size=1)
    X = X - np.mean(X, axis=0)
    X = torch.from_numpy(X).to(device, dtype=torch.float32)
    X_c = (X * torch.cat([torch.ones((size,1))]+[torch.zeros((size,1))]*(NODES-1), dim=1).to(device, dtype=torch.float32))
    update_mask = torch.cat([torch.zeros((size,1))]+[torch.ones((size,1))]*(NODES-1), dim=1)
    return X, X_c, update_mask

def get_gaussian(size):
    mean = np.array([0 ,0])
    cov = np.array([[1, 0.5],
                    [0.5, 1]])
    X = np.empty((size, 2))
    np.random.seed(0)              
    for i in range(size):
        X[i] = np.random.multivariate_normal(mean, cov, size=1)
    X = torch.from_numpy(X).to(device, dtype=torch.float32)
    X_c = (X * torch.cat([torch.ones((size,1))]+[torch.zeros((size,1))]*(NODES-1), dim=1).to(device, dtype=torch.float32))
    update_mask = torch.cat([torch.zeros((size,1))]+[torch.ones((size,1))]*(NODES-1), dim=1)
    return X, X_c, update_mask


def memory_ML(X, X_c, update_mask, training_lr, training_iters, relaxation_lr, relaxation_iters):
    """relaxation using the maximum likelihood estimate of the covariance matrix"""
    ML_cov = torch.cov(X.T)
    X_recon_ML = X_c.clone()
    for i in range(relaxation_iters):
        print('iteration:', i)
        errs_ML = torch.matmul(X_recon_ML, torch.linalg.inv(ML_cov).T) 
        print(torch.mean(errs_ML**2))
        delta = -errs_ML
        X_recon_ML += relaxation_lr * (delta * update_mask)
    return X_recon_ML


def memory_Friston(X, X_c, update_mask, training_lr, training_iters, relaxation_lr, relaxation_iters):
    """relaxation using Friston's update of covariance matrix"""
    ## learning the covariance
    F_cov = torch.eye(NODES)  # initial cov
    mse_F = []
    for i in range(training_iters):
        for batch_idx in range(0, sample_size, batch_size):
            batchX = X[batch_idx:batch_idx+batch_size]
            errsW_F = torch.matmul(batchX, torch.linalg.inv(F_cov).T) # N, 2
            grad_cov = 0.5 * (torch.matmul(errsW_F.T, errsW_F)/batch_size - torch.linalg.inv(F_cov))
            F_cov += training_lr * grad_cov

        ## relaxation using the current covariance
        X_recon_F = X_c
        for i in range(relaxation_iters):
            errsX_F = torch.matmul(X_recon_F, torch.linalg.inv(F_cov).T) 
            delta = -errsX_F
            X_recon_F += relaxation_lr * (delta * update_mask)
        mse_F.append(torch.mean((X[:,1] - X_recon_F[:,1])**2)) 
    print(sum(mse_F))
    print('Friston:', F_cov)

    return X_recon_F, mse_F

def memory_Rafal(X, X_c, update_mask, training_lr, training_iters, relaxation_lr, relaxation_iters):
    """relaxation using Rafal's update of covariance matrix"""
    ## learning the covariance
    R_cov = torch.eye(NODES)
    mse_R = []
    for i in range(training_iters):
        for batch_idx in range(0, sample_size, batch_size):
            batchX = X[batch_idx:batch_idx+batch_size]
            errsW_R = torch.matmul(batchX, torch.linalg.inv(R_cov).T) # N, 2
            e = batchX
            delta = torch.matmul(errsW_R.T, e)/batch_size - torch.eye(NODES)
            R_cov += training_lr * delta

        ## relaxation using the current covariance
        X_recon_R = X_c
        for i in range(relaxation_iters):
            errsX_R = torch.matmul(X_recon_R, torch.linalg.inv(R_cov).T) 
            delta = -errsX_R
            X_recon_R += relaxation_lr * (delta * update_mask)
        mse_R.append(torch.mean((X[:,1] - X_recon_R[:,1])**2)) 
    print(sum(mse_R))
    print('Rafal:', R_cov)

    return X_recon_R, mse_R

def memory_rec(X, X_c, update_mask, training_lr, training_iters, relaxation_lr, relaxation_iters, backward=False):
    """relaxation using the learned weights by recurrent PCN"""

    ## learning the weights
    bound = 1./np.sqrt(NODES)
    init_weights = torch.zeros((NODES, NODES))
    weights = init_weights.fill_diagonal_(0).to(device)
    mse_rec = []
    for i in range(training_iters):
        # print('iter:', i)
        for batch_idx in range(0, sample_size, batch_size):
            batchX = X[batch_idx:batch_idx+batch_size]
            preds = torch.matmul(batchX, weights.t())
            errsW_rec = batchX - preds
            grad_w = torch.matmul(errsW_rec.T, batchX).fill_diagonal_(0)
            weights += training_lr * grad_w
        
        ## relaxation using the current weights
        X_recon_rec = X_c.clone()
        for i in range(relaxation_iters):
            errsX_rec = X_recon_rec - torch.matmul(X_recon_rec, weights.T)
            if backward:
                delta = -errsX_rec + torch.matmul(errsX_rec, weights)
            else:
                delta = -errsX_rec
            X_recon_rec += relaxation_lr * (delta * update_mask)
        mse_rec.append(torch.mean((X - X_recon_rec)**2))

    return X_recon_rec, mse_rec

def main():
    rep = 10
    training_lr = 0.001
    training_iters = 100
    relaxation_lr = 0.01
    relaxation_iters = 100
    X, X_c, update_mask = get_gaussian(sample_size)
    X_recon_rec, mse_rec = memory_rec(X, X_c, update_mask, training_lr, training_iters, relaxation_lr, relaxation_iters)
    X_recon_F, mse_F = memory_Friston(X, X_c, update_mask, training_lr, training_iters, relaxation_lr, relaxation_iters)
    plt.figure()
    plt.plot(mse_rec)
    plt.plot(mse_F)
    plt.show()
    
  

if __name__ == '__main__':
    main()