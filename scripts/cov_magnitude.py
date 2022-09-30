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

model_path = './models/'
n_layers = [1, 2, 3, 4]

inference_lr = 1e-2
dendritic = True
seeds = range(5)
sample_size = 256
dataset = 'cifar10'

hy_distances = np.zeros((len(n_layers), len(seeds)))
for j, seed in enumerate(seeds):
    for i, n_layer in enumerate(n_layers):
        nodes = [256] * n_layer + [1024]
        pcn_h = HybridPCN(nodes, 'Tanh', inference_lr, dendritic=dendritic).to(device)
        state_dict = torch.load(model_path+f'n_layers{n_layer}_dendritic{dendritic}_pcn_seed{seed}_sample_size{sample_size}_{dataset}.pt', map_location=torch.device('cpu'))
        top_activities = state_dict['top_activities'].cpu().detach().numpy()
        S = np.cov(top_activities.T)
        # measuring the distance between the off diagonal elements of S and 0
        distance = np.sqrt(np.sum(S ** 2) - np.sum(S.diagonal() ** 2))
        hy_distances[i, j] = distance

data_distances = np.zeros((1, len(seeds)))
for seed in seeds:
    (X, _), (X_test, _) = get_cifar10('./data', 
                                    sample_size=sample_size, 
                                    sample_size_test=1,
                                    batch_size=1, 
                                    seed=seed, 
                                    device=device,
                                    classes=None)
    size = X.shape
    flattened_size = size[-1]*size[-2]*size[-3]
    X = X.reshape(-1, flattened_size) 
    data = X[-(sample_size // 8):]
    S_data = np.cov(data.T)
    distance = np.sqrt(np.sum(S_data ** 2) - np.sum(S_data.diagonal() ** 2))
    data_distances[:, seed] = distance

distances = np.log2(np.concatenate((data_distances, hy_distances), axis=0))
x_pos = range(len(n_layers)+1)
plt.figure()
plt.bar(x_pos, np.mean(distances, axis=1), yerr=np.std(distances, axis=1), align='center', alpha=0.5, ecolor='black')
plt.xticks(x_pos)
plt.xlabel('Number of layers')
plt.ylabel('Magnitude of covariances (log-scaled)')
plt.savefig('./results/distances', dpi=300)

