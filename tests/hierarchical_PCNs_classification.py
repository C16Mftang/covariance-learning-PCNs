"""
Investigation of a two-layer latent predictive coding net, with or without recurrent connections, 
in terms of their capability of feature extractions.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

latent_dim = 100
dim = 784
sample_size = 5000
test_size = 1000

X = np.load('./data/MNIST.npy')
y = np.load('./data/MNIST_labels.npy')

X_tr, X_test = X[0:sample_size], X[sample_size:sample_size+test_size]
y_tr, y_test = y[0:sample_size], y[sample_size:sample_size+test_size]

class PCN(object):
    def __init__(self, dim, latent_dim, recurrent=False):
        self.dim = dim
        self.latent_dim = latent_dim
        self.recurrent = recurrent

        # initialize parameters
        self.curr_mu_z = np.random.randn(latent_dim)
        self.curr_W = np.random.randn(dim, latent_dim) # k*d
        self.curr_z = np.random.randn(latent_dim) # 1*d
        if recurrent:
            self.curr_Wr = np.zeros((dim, dim))
            np.fill_diagonal(self.curr_Wr, 0)

    def relaxation(self, X, relax_itrs, lr_z):
        rec_pred = np.matmul(X, self.curr_Wr.T) if self.recurrent else 0
        pred_X = np.matmul(self.curr_z, self.curr_W.T) + rec_pred # 1*k
        self.err_X = X - pred_X # bsz*k
        self.err_z = self.curr_z - self.curr_mu_z # 1*d
        for j in range(relax_itrs):
            delta_z = (-self.err_z + np.matmul(self.err_X, self.curr_W)) # bsz*d
            self.curr_z = self.curr_z + lr_z * delta_z # bsz*d
            # update error nodes
            pred_X = np.matmul(self.curr_z, self.curr_W.T) + rec_pred # bsz*k
            self.err_X = X - pred_X # bsz*k
            self.err_z = self.curr_z - self.curr_mu_z # 1*d

    def learning(self, lr_W, lr_Wr, X):
        delta_W = np.matmul(self.err_X.T, self.curr_z) # k*d
        self.curr_W = self.curr_W + lr_W * delta_W
        delta_mu_z = np.sum(self.err_z, axis=0) # 1*d
        self.curr_mu_z = self.curr_mu_z + lr_W * delta_mu_z
        if self.recurrent:
            delta_Wr = np.matmul(self.err_X.T, X) / 1e4
            np.fill_diagonal(delta_Wr, 0)
            self.curr_Wr = self.curr_Wr + lr_Wr * delta_Wr
            

    def training(self, X, lr_z, lr_W, lr_Wr, relax_itrs, epochs, batch_size):
        err_Xs = []
        for e in range(epochs):
            curr_zs = np.empty((X.shape[0], self.latent_dim))
            for i in range(0, X.shape[0], batch_size):
                # set nodes given the current batch of observations
                batch_X = X[i:i+batch_size] # bsz*k
                
                # relaxation/inference
                self.relaxation(batch_X, relax_itrs, lr_z)

                # learning
                self.learning(lr_W, lr_Wr, batch_X)

                curr_zs[i:i+batch_size] = self.curr_z # N*d
            
            if e % 1 == 0:
                print(f'Epoch {e} prediction error (MSE):')
                print(np.mean(self.err_X**2))
                # print(np.mean(self.curr_Wr**2))
                # print(np.mean(np.matmul(batch_X, self.curr_Wr.T)**2))
            rec_pred = np.matmul(X, self.curr_Wr.T) if self.recurrent else 0     
            all_pred = np.matmul(curr_zs, self.curr_W.T) + rec_pred
            all_err_X = X - all_pred
            err_Xs.append(np.mean(all_err_X**2))

        return err_Xs, curr_zs

    def relax_on_test(self, X_test, relax_itrs, lr_z, batch_size):
        # update error nodes given the test set
        z = np.random.randn(latent_dim) # 1*d, re-initialize the latent variable
        rec_pred = np.matmul(X_test, self.curr_Wr.T) if self.recurrent else 0     
        pred_X = np.matmul(z, self.curr_W.T) + rec_pred # 1*k
        err_X = X_test - pred_X # bsz*k
        err_z = z - self.curr_mu_z # 1*d
        for j in range(relax_itrs):
            delta_z = (-err_z + np.matmul(err_X, self.curr_W)) # bsz*d
            z = z + lr_z * delta_z # bsz*d
            # update error nodes
            pred_X = np.matmul(z, self.curr_W.T) + rec_pred # bsz*k
            err_X = X_test - pred_X # bsz*k
            err_z = z - self.curr_mu_z # 1*d

        return z

# main
pcn = PCN(dim=784, latent_dim=100, recurrent=True)
epochs = 200
batch_size = 100
relax_itrs = 100
lr_z = 3e-5
lr_W = 1e-5
lr_Wr = 1e-6
err_Xs, curr_zs = pcn.training(X_tr, lr_z, lr_W, lr_Wr, relax_itrs, epochs, batch_size)
z = pcn.relax_on_test(X_test, relax_itrs, lr_z, batch_size)

clf = LogisticRegression(max_iter=2000, solver='sag').fit(curr_zs, y_tr)
print(clf.score(z, y_test))