import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('seaborn')
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from src.get_data import *
from src.models import *
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# hyperparameters
# vary these hyparams for different models and datasets
model_type = 'imp' # 'exp', 'imp', 'den'
nonlin = 'linear' # 'linear', 'rate', 'tanh', 'binary'
dataset = 'cifar10' # 'cifar10' and 'mnist'
corruption = 'cover' # 'cover' or 'noise'

# sample_sizes = [8, 16, 32, 64, 128, 256] # for full reproduction
sample_sizes = [16, 128]
sample_size_test = 1
batch_sizes = [sample_size // 8 for sample_size in sample_sizes]
noise = 0.05
divisor = 2
inference_lr = 0.1
learning_lr = 1e-4 # 1e-5 for explicit model
learning_iters = [200, 400]
seeds = range(3)
image_size = 32
model_path = './models/'
result_path = os.path.join('./results/', 'novelty_detection')
if not os.path.exists(result_path):
    os.makedirs(result_path)

all_mses = np.zeros((len(sample_sizes), len(seeds)))
for k in range(len(sample_sizes)):
    sample_size, batch_size, learning_iter = sample_sizes[k], batch_sizes[k], learning_iters[k]
    for ind, seed in enumerate(seeds):
        print(f'sample size {sample_size}, seed {seed}')
        sub_path = os.path.join(result_path, f'{sample_size}_samples_seed_{seed}')
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        (X, _), (X_test, _) = get_cifar10('./data', 
                                            sample_size=sample_size, 
                                            sample_size_test=sample_size,
                                            batch_size=batch_size, 
                                            seed=seed, 
                                            device=device,
                                            classes=None)
        size = X.shape
        flattened_size = size[-1]*size[-2]*size[-3]
        X = X.reshape(-1, flattened_size)
        X_test = X_test.reshape(-1, flattened_size)
        print(flattened_size)

        if model_type == 'exp':
            pcn = ExplicitPCN(flattened_size).to(device)
        elif model_type == 'den':
            pcn = RecPCN(flattened_size, dendrite=True, mode=nonlin).to(device)
        elif model_type == 'imp':
            pcn = RecPCN(flattened_size, dendrite=False, mode=nonlin).to(device)
        else:
            raise ValueError("no such model type")
        optimizer = torch.optim.Adam(pcn.parameters(), lr=learning_lr)

        # training
        print(f'Training for {learning_iter} epochs')
        train_mses = []
        for i in range(learning_iter):
            for batch_idx in range(0, sample_size, batch_size):
                data = X[batch_idx:batch_idx+batch_size]
                optimizer.zero_grad()
                pcn.learning(data)
                optimizer.step()
            train_mse = pcn.train_mse.cpu().detach().numpy()
            train_mses.append(train_mse)

            if i % 10 == 0:
                print(f'Epoch {i}, mse {train_mse}')

        energy_fam = pcn.energy(X).cpu().detach().numpy()
        energy_nov = pcn.energy(X_test).cpu().detach().numpy()
        
        plt.figure()
        plt.hist(energy_fam, label='familiar')
        plt.hist(energy_nov, label='novel')
        plt.legend()
        plt.savefig(sub_path+'/energy_distributions', dpi=150)
