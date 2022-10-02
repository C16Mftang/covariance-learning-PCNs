import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from src.get_data import *
from src.models import *
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
plt.style.use('seaborn')

# hyperparameters
learning_iters = 200
learning_lr = 5e-4
inference_iters = 400
inference_lr = 1e-2
noise_var = 0.05
divisor = 2
image_size = 32
sample_sizes = [256]
sample_size_test = 1
batch_sizes = [sample_size // 8 for sample_size in sample_sizes]
seeds = range(3)
n_layers = [1, 2, 3, 4]
model_path = './models/'
result_path = './results/'
dataset = 'cifar10'
dendritic = True

# training
for n_layer in n_layers:
    for k in range(len(sample_sizes)):
        sample_size, batch_size = sample_sizes[k], batch_sizes[k]
        for seed in seeds:
            print(f'sample size {sample_size}, seed {seed}, layers {n_layer}')
            (X, _), (X_test, _) = get_cifar10('./data', 
                                                sample_size=sample_size, 
                                                sample_size_test=sample_size_test,
                                                batch_size=batch_size, 
                                                seed=seed, 
                                                device=device,
                                                classes=None)
            size = X.shape
            flattened_size = size[-1]*size[-2]*size[-3]
            X_c, update_mask = cover_bottom(X, divisor, device)
            X = X.reshape(-1, flattened_size)
            print(flattened_size)

            nodes = [256] * n_layer + [flattened_size]
            pcn_h = HybridPCN(nodes, 'Tanh', inference_lr, dendritic=dendritic, init_std=1.).to(device)

            optimizer = torch.optim.Adam(pcn_h.parameters(), lr=learning_lr)

            ########################################################################
            # train model
            train_mses = []
            for i in range(learning_iters):
                print('Epoch', i)
                with tqdm(total=sample_size) as progress_bar:
                    for batch_idx in range(0, sample_size, batch_size):
                        data = X[batch_idx:batch_idx+batch_size]
                        optimizer.zero_grad()
                        pcn_h.train_pc_generative(data, inference_iters, update_mask)
                        optimizer.step()
                        progress_bar.update(batch_size)
                train_mse = torch.mean(pcn_h.errs[-1]**2).cpu().detach().numpy()
                train_mses.append(train_mse)

            torch.save(pcn_h.state_dict(), 
                       model_path+f'n_layers{n_layer}_dendritic{dendritic}_pcn_seed{seed}_sample_size{sample_size}_{dataset}_initstd{1.0}.pt')

            # plot
            # plt.rcParams['figure.dpi'] = 70
            # plt.figure()
            # plt.plot(train_mses)
            # plt.show()

