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
import argparse
from torch.distributions.multivariate_normal import MultivariateNormal
from src.get_data import *
from src.models import *
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_type = 'hn'
dataset = "bmnist"  # 'cifar10' and 'mnist' and 'bmnist'
corruption = "cover"  # 'cover' or 'noise'

# sample_sizes = [8, 16, 32, 64, 128, 256] # for full reproduction
sample_sizes = [2,4,6,8,10,12]
# sample_sizes = [2]
sample_size_test = 1
batch_sizes = [sample_size // 1 for sample_size in sample_sizes]
noise = 0.05
divisor = 2
inference_lr = 0.1

# inference_iters = [2000, 2000, 2000, 2000, 5000, 5000] # for full reproduction
inference_iters = [10] * len(sample_sizes)

seeds = [0, 1, 3, 4, 120, 1200, 10000]
# seeds = [0]
image_size = 28
model_path = "./models/"
result_path = os.path.join(
    "./results/", f"{model_type}_{dataset}_{corruption}"
)
if not os.path.exists(result_path):
    os.makedirs(result_path)

all_mses = np.zeros((len(sample_sizes), len(seeds)))
for k in range(len(sample_sizes)):
    sample_size, batch_size, inference_iter = (
        sample_sizes[k],
        batch_sizes[k],
        inference_iters[k],
    )
    for ind, seed in enumerate(seeds):
        print(f"sample size {sample_size}, seed {seed}")
        sub_path = os.path.join(result_path, f"{sample_size}_samples_seed_{seed}")
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        (X, _), (X_test, _) = get_mnist(
            "./data",
            sample_size=sample_size,
            sample_size_test=sample_size_test,
            batch_size=batch_size,
            seed=seed,
            device=device,
            classes=None,
            binary=True
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

        hn = HopfieldNetwork(dim=flattened_size).to(device)
        hn.learning(X)

        inference_mses_hn = []
        X_recon_hn = X_c.clone()
        for j in range(inference_iter):
            X_recon_hn = hn.inference(X_recon_hn)
            mse = torch.mean((X_recon_hn - X) ** 2).cpu().detach().numpy()
            inference_mses_hn.append(mse)
        all_mses[k, ind] = inference_mses_hn[-1]

        # visualization of current sample and seed and save figure
        plt.rcParams["figure.dpi"] = 70
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(inference_mses_hn)
        ax[0].set_title("retrieval MSE")
        # ax[1].plot(train_mses)
        # ax[1].set_title("train MSE")
        plt.savefig(sub_path + "/MSEs")

        fig, ax = plt.subplots(3, 2, figsize=(2, 4))
        for i in range(2):
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
                X_recon_hn[i].reshape((image_size, image_size)).cpu().detach().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            ax[2, i].axis("off")
            ax[2, 0].set_title(f"{model_type}")
        plt.savefig(sub_path + "/examples", bbox_inches='tight')

# finally save the retrieval MSEs for reproducing the figures
print("retrieval mses for all sample sizes and seeds")
print(all_mses)
np.save(result_path + "/retrieval_MSEs", all_mses)
