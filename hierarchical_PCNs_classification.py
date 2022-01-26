"""
Investigation of a two-layer latent predictive coding net, with or without recurrent connections, 
in terms of their capability of feature extractions.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 100
dim = 784
sample_size = 50000

X = np.load('./data/MNIST.npy')
y = np.load('./data/MNIST_labels.npy')

X_tr, X_test = X[0:sample_size], X[sample_size:]
y_tr, y_test = y[0:sample_size], y[sample_size:]

def standard_pcn(X):
    # The STANDARD model; d: dimension of latent; k: dimension of observed
    # X: observed, z: latent
    epochs = 500
    batch_size = 100
    relax_itrs = 100
    lr_z = 3e-3
    lr_W = 1e-3

    # initialize parameters
    curr_mu_z = np.zeros((latent_dim,))
    curr_W = np.zeros((dim, latent_dim)) # k*d
    err_Xs = []
    curr_z = np.zeros_like(curr_mu_z) # 1*d

    for e in range(epochs):
        curr_zs = np.empty((X.shape[0], latent_dim))
        for i in range(0, X.shape[0], batch_size):
            # set nodes given the current batch of observations
            batch_X = X[i:i+batch_size] # bsz*k
            pred_X = np.matmul(curr_z, curr_W.T) # 1*k
            err_X = batch_X - pred_X # bsz*k
            err_z = curr_z - curr_mu_z # 1*d
            
            # relaxation
            for j in range(relax_itrs):
                delta_z = (-err_z + np.matmul(err_X, curr_W)) / batch_size # bsz*d
                curr_z = curr_z + lr_z * delta_z # bsz*d
                # update error nodes
                pred_X = np.matmul(curr_z, curr_W.T) # bsz*k
                err_X = batch_X - pred_X # bsz*k
                err_z = curr_z - curr_mu_z # 1*d

            # learning
            delta_W = np.matmul(err_X.T, curr_z) / batch_size # k*d
            curr_W = curr_W + lr_W * delta_W
            delta_mu_z = np.sum(err_z, axis=0) / batch_size # 1*d
            curr_mu_z = curr_mu_z + lr_W * delta_mu_z

            curr_zs[i:i+batch_size] = curr_z # N*d
        
        if e % 10 == 0:
            print(f'Epoch {e} prediction error (MSE):')
            print(np.mean(err_X**2))
        all_pred = np.matmul(curr_zs, curr_W.T)
        all_err_X = X - all_pred
        err_Xs.append(np.mean(all_err_X**2))

    return err_Xs, curr_zs

err_Xs, curr_zs = standard_pcn(X_tr)
plt.figure()
plt.plot(err_Xs)
plt.show()
