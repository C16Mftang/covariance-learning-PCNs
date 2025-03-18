"""
This script trains single-layer PCNs for associatiev memory tasks, and saves the models to the models folder

It can be used to produce figures 4 and 6 in the paper
"""

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

plt.style.use("seaborn")
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from src.get_data import *
from src.models import *
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# hyperparameters
# vary these hyparams for different models and datasets
model_type = "imp"  # 'exp', 'imp', 'den'
nonlin = "linear"  # 'linear', 'rate', 'tanh', 'binary'
dataset = "cifar10"  # 'cifar10' and 'mnist'
corruption = "cover"  # 'cover' or 'noise'

# sample_sizes = [8, 16, 32, 64, 128, 256] # for full reproduction
sample_sizes = [64]
sample_size_test = 1
batch_sizes = [sample_size // 8 for sample_size in sample_sizes]
noise = 0.05
divisor = 2
inference_lr = 0.1
if model_type == "exp":
    # inference_iters = [2000, 2000, 2000, 2000, 5000, 5000] # for full reproduction
    inference_iters = [2000]
else:
    # inference_iters = [10000, 10000, 10000, 50000, 100000, 100000] # for full reproduction
    inference_iters = [50000]
learning_lr = 1e-4  # 1e-5 for explicit model
# learning_iters = [200, 200, 200, 400, 400, 400] # for full reproduction
learning_iters = [400]
# seeds = [0, 1, 3, 4, 120, 1200, 10000]
seeds = range(2)
image_size = 32
model_path = "./models/"
result_path = os.path.join(
    "./results/", f"{model_type}_{nonlin}_{dataset}_{corruption}"
)
if not os.path.exists(result_path):
    os.makedirs(result_path)

all_mses = np.zeros((len(sample_sizes), len(seeds)))
for k in range(len(sample_sizes)):
    sample_size, batch_size, learning_iter, inference_iter = (
        sample_sizes[k],
        batch_sizes[k],
        learning_iters[k],
        inference_iters[k],
    )
    for ind, seed in enumerate(seeds):
        print(f"sample size {sample_size}, seed {seed}")
        sub_path = os.path.join(result_path, f"{sample_size}_samples_seed_{seed}")
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        (X, _), (X_test, _) = get_cifar10(
            "./data",
            sample_size=sample_size,
            sample_size_test=sample_size_test,
            batch_size=batch_size,
            seed=seed,
            device=device,
            classes=None,
        )
        size = X.shape
        flattened_size = size[-1] * size[-2] * size[-3]

        # make the corrupted data
        if corruption == "cover":
            X_c, update_mask = cover_bottom(X, divisor, device)
        elif corruption == "noise":
            X_c, update_mask = add_gaussian_noise(X, noise, device)
        else:
            raise ValueError("no such corruption type")
        X = X.reshape(-1, flattened_size)
        print(flattened_size)

        if model_type == "exp":
            pcn = ExplicitPCN(flattened_size).to(device)
        elif model_type == "den":
            pcn = RecPCN(flattened_size, dendrite=True, mode=nonlin).to(device)
        elif model_type == "imp":
            pcn = RecPCN(flattened_size, dendrite=False, mode=nonlin).to(device)
        else:
            raise ValueError("no such model type")
        optimizer = torch.optim.Adam(pcn.parameters(), lr=learning_lr)

        # training
        print(f"Training for {learning_iter} epochs")
        train_mses = []
        for i in range(learning_iter):
            for batch_idx in range(0, sample_size, batch_size):
                data = X[batch_idx : batch_idx + batch_size]
                optimizer.zero_grad()
                pcn.learning(data)
                optimizer.step()
            train_mse = pcn.train_mse.cpu().detach().numpy()
            train_mses.append(train_mse)

            if i % 10 == 0:
                print(f"Epoch {i}, mse {train_mse}")

        torch.save(pcn.state_dict(), sub_path + "/model.pt")

        # retrieval
        print(f"Perform retrieval for {inference_iter} iterations")
        X_recon = X_c.clone()
        retrieval_mses = []
        deltas = []
        with torch.no_grad():
            for itr in range(inference_iter):
                delta_X = pcn.inference(X_recon)
                X_recon += inference_lr * (delta_X * update_mask)

                retrieval_mse = torch.mean((X_recon - X) ** 2)
                retrieval_mses.append(retrieval_mse.cpu().detach().numpy())
                delta = torch.mean(delta_X ** 2)
                deltas.append(delta.cpu().detach().numpy())

                if itr % (inference_iter // 10) == 0:
                    print(f"iter {itr}, retrieval mse {retrieval_mse}")
        all_mses[k, ind] = retrieval_mses[-1]

        # visualization of current sample and seed and save figure
        plt.rcParams["figure.dpi"] = 70
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(retrieval_mses)
        ax[0].set_title("retrieval MSE")
        ax[1].plot(train_mses)
        ax[1].set_title("train MSE")
        plt.savefig(sub_path + "/MSEs")

        fig, ax = plt.subplots(3, 5, figsize=(7, 5))
        for i in range(5):
            ax[0, i].imshow(
                X[i].reshape((image_size, image_size)).cpu().detach().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            ax[0, i].axis("off")
            ax[1, i].imshow(
                X_c[i].reshape((image_size, image_size)).cpu().detach().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            ax[1, i].axis("off")
            ax[2, i].imshow(
                X_recon[i].reshape((image_size, image_size)).cpu().detach().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            ax[2, i].axis("off")
            ax[2, 0].set_title(f"{model_type}")
        plt.savefig(sub_path + "/examples")

# finally save the retrieval MSEs for reproducing the figures
print("retrieval mses for all sample sizes and seeds")
print(all_mses)
np.save(result_path + "/retrieval_MSEs", all_mses)
