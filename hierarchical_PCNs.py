"""
Code for investigating the effect of recurrent connections in a two-layer hierarchical PCN
i.e. a latent variable model.

Laten variable: x, observations $y \sim N(Wx, \Sigma)$. Goal is to find the most likely x and model parameter W
"""

import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

nonlinear = False
def nonlinearity(x):
    if nonlinear:
        return np.tanh(x)
    else:
        return x

def deriv(x):
    if nonlinear:
        return 1.0 - np.tanh(x) ** 2.0
    else:
        return 1.0
    
# Generate observed data. One latent variable for each observation
sample_size = 1000
W = np.array([[2, 1],
              [1, 2]])
mu_x = np.array([1, 1])
Sx = np.eye(2)

def data_generation(cov):
    Sy = np.array([[1, cov],
                   [cov, 1]])           
    x = np.empty((sample_size, 2))
    y = np.empty((sample_size, 2))
    mu_y = np.empty((sample_size, 2))

    np.random.seed(48)
    for i in range(sample_size):
        xi = np.random.multivariate_normal(mu_x, Sx)
        mu_yi = np.matmul(W, xi)
        yi = np.random.multivariate_normal(mu_yi, Sy)
        x[i] = xi
        y[i] = yi
        mu_y[i] = mu_yi

    # y_c = y * np.concatenate([np.ones((sample_size,1)), np.ones((sample_size,1))], axis=1)
    return y

def standard_pcn(y):
    # The STANDARD model; d: dimension of x; k: dimension of y
    epochs = 200
    batch_size = 100
    relax_itrs = 100
    eval_itrs = 100
    lr_x = 3e-3
    lr_W = 1e-3
    lr_eval = 3e-3
    # initialize parameters
    curr_mu_x = np.zeros_like(mu_x)
    curr_W = np.eye(2) # k*d
    std_mses_mu_x = []
    err_ys = []

    for e in range(epochs):
        for i in range(0, sample_size, batch_size):
            # set nodes given the current batch of observations
            curr_x = np.copy(curr_mu_x) # 1*d
            batch_y = y[i:i+batch_size] # bsz*k
            pred_y = np.matmul(curr_x, curr_W.T) # 1*k
            err_y = batch_y - pred_y # bsz*k
            err_x = curr_x - curr_mu_x # 1*d
            
            # relaxation
            for j in range(relax_itrs):
                delta_x = (-err_x + np.matmul(err_y, curr_W)) / batch_size # bsz*d
                curr_x = curr_x + lr_x * delta_x # bsz*d
                # update error nodes
                pred_y = np.matmul(curr_x, curr_W.T) # bsz*k
                err_y = batch_y - pred_y # bsz*k
                err_x = curr_x - curr_mu_x # bsz*d

            # learning
            delta_W = np.matmul(err_y.T, curr_x) / batch_size # k*d
            curr_W = curr_W + lr_W * delta_W
            delta_mu_x = np.sum(err_x, axis=0) # 1*d
            curr_mu_x = curr_mu_x + lr_W * delta_mu_x
        
        err_ys.append(np.mean(err_y**2))
        std_mses_mu_x.append(np.mean((curr_mu_x - mu_x)**2))
        
    print('Standard PCN')
    print(curr_mu_x)
    print(curr_W)

    return err_ys


def recurrent_pcn(y):

    # The RECURRENT model; d: dimension of x; k: dimension of y
    epochs = 200
    batch_size = 100
    relax_itrs = 100
    eval_itrs = 100
    lr_x = 3e-3
    lr_W = 1e-3
    lr_Wr = 5e-5
    lr_eval = 3e-3
    # initialize parameters
    curr_mu_x = np.zeros_like(mu_x)
    curr_W = np.eye(2) # k*d
    curr_Wr = np.zeros((2, 2))
    rec_mses_mu_x = []
    err_ys = []

    for e in range(epochs):
        for i in range(0, sample_size, batch_size):
            # set nodes given the current batch of observations
            curr_x = np.copy(curr_mu_x) # 1*d
            batch_y = y[i:i+batch_size] # bsz*k
            pred_y = np.matmul(curr_x, curr_W.T) + np.matmul(batch_y, curr_Wr) # 1*k
            err_y = batch_y - pred_y # bsz*k
            err_x = curr_x - curr_mu_x # 1*d
            
            # relaxation
            for j in range(relax_itrs):
                delta_x = (-err_x + np.matmul(err_y, curr_W)) / batch_size # bsz*d
                curr_x = curr_x + lr_x * delta_x # bsz*d
                # update error nodes
                pred_y = np.matmul(curr_x, curr_W.T) + np.matmul(batch_y, curr_Wr) # bsz*k
                err_y = batch_y - pred_y # bsz*k
                err_x = curr_x - curr_mu_x # bsz*d

            # learning
            delta_W = np.matmul(err_y.T, curr_x) / batch_size # k*d
            curr_W = curr_W + lr_W * delta_W
            delta_Wr = np.matmul(err_y.T, batch_y) / batch_size # k*k
            np.fill_diagonal(delta_Wr, 0)
            curr_Wr = curr_Wr + lr_Wr * delta_Wr
            delta_mu_x = np.sum(err_x, axis=0) # 1*d
            curr_mu_x = curr_mu_x + lr_W * delta_mu_x
        
        err_ys.append(np.mean(err_y**2))
        rec_mses_mu_x.append(np.mean((curr_mu_x - mu_x)**2))
        
    print('Recurrent PCN')
    print(curr_mu_x)
    print(curr_W)
    print(curr_Wr)

    return err_ys

fig, ax = plt.subplots(1, 2, figsize=(10,4))
for cov in [0, 0.1, 1]:
    y = data_generation(cov)
    std_err_ys = standard_pcn(y)
    rec_err_ys = recurrent_pcn(y)
    ax[0].plot(std_err_ys, label=f'cov={cov}')
    ax[1].plot(rec_err_ys, label=f'cov={cov}')
ax[0].legend()
ax[0].set_title('squared error in y: standard')
ax[0].set_xlabel('training epochs')
ax[1].legend()
ax[1].set_title('squared error in y: recurrent')
ax[1].set_xlabel('training epochs')
plt.show()












        
