"""
This script trains single-layer PCNs for associatiev memory tasks, and saves the models to the models folder

It can be used to produce figures 3 in the paper
"""

import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")
from src.get_data import *
from src.models import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

learning_iters = 200
learning_lr = 5e-3
inference_iters = 100
inference_lr = 0.1
batch_size = 20
cover_size = 12
image_size = 5
sample_size = 20
cover_sizes = np.arange(2, 12, 2)
seeds = np.arange(1)
binary = True
# create save path
if binary:
    result_path = os.path.join("./results/", "25d_binary")
else:
    result_path = os.path.join("./results/", "25d_random")
if not os.path.exists(result_path):
    os.makedirs(result_path)

explicit_mses = np.zeros((len(cover_sizes), len(seeds)))
implicit_mses = np.zeros((len(cover_sizes), len(seeds)))
dendritic_mses = np.zeros((len(cover_sizes), len(seeds)))
hn_mses = np.zeros((len(cover_sizes), len(seeds)))
for c, cover_size in enumerate(cover_sizes):
    for s, seed in enumerate(seeds):
        print(f"cover size: {cover_size}; seed: {seed}")
        sub_path = os.path.join(result_path, f"cover_{cover_size}_seed_{seed}")
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        X = get_gaussian(
            "./data",
            sample_size=sample_size,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )

        if binary:
            X = (X > 0).float()

        # corrupt the pattern by setting the last [cover_size] dimensions to 0
        X = X.reshape(-1, image_size * image_size)
        mask = torch.ones_like(X).to(device)
        mask[:, -cover_size:] -= 1
        X_c = (X * mask).to(device)
        # only update the corrupted dimensions
        update_mask = torch.zeros_like(X).to(device)
        update_mask[:, -cover_size:] += 1

        hn = HopfieldNetwork(dim=image_size ** 2).to(device)
        hn.learning(X)

        inference_mses_hn = []
        X_recon_hn = X_c.clone()
        for j in range(inference_iters):
            X_recon_hn = hn.inference(X_recon_hn)
            mse = torch.mean((X_recon_hn - X) ** 2).cpu().detach().numpy()
            inference_mses_hn.append(mse)
        hn_mses[c, s] = inference_mses_hn[-1]

        # explicit model
        pcn_exp = ExplicitPCN(dim=image_size ** 2).to(device)
        optimizer = torch.optim.SGD(pcn_exp.parameters(), lr=learning_lr)
        learning_mses_exp = []
        for i in range(learning_iters):
            for batch_idx in range(0, sample_size, batch_size):
                data = X[batch_idx : batch_idx + batch_size]
                optimizer.zero_grad()
                pcn_exp.learning(data)
                optimizer.step()
            mse = pcn_exp.train_mse.cpu().detach().numpy()
            learning_mses_exp.append(mse)

        inference_mses_exp = []
        X_recon_exp = X_c.clone()
        for j in range(inference_iters):
            delta_X = pcn_exp.inference(X_recon_exp)
            X_recon_exp += inference_lr * (delta_X * update_mask)
            mse = torch.mean((X_recon_exp - X) ** 2).cpu().detach().numpy()
            inference_mses_exp.append(mse)

        explicit_mses[c, s] = inference_mses_exp[-1]

        # implicit model
        pcn_imp = RecPCN(dim=image_size ** 2, dendrite=False).to(device)
        optimizer = torch.optim.SGD(pcn_imp.parameters(), lr=learning_lr)
        learning_mses_imp = []
        for i in range(learning_iters):
            for batch_idx in range(0, sample_size, batch_size):
                data = X[batch_idx : batch_idx + batch_size]
                optimizer.zero_grad()
                pcn_imp.learning(data)
                optimizer.step()
            mse = pcn_imp.train_mse.cpu().detach().numpy()
            learning_mses_imp.append(mse)

        inference_mses_imp = []
        X_recon_imp = X_c.clone()
        for j in range(inference_iters):
            delta_X = pcn_imp.inference(X_recon_imp)
            X_recon_imp += inference_lr * (delta_X * update_mask)
            mse = torch.mean((X_recon_imp - X) ** 2).cpu().detach().numpy()
            inference_mses_imp.append(mse)

        implicit_mses[c, s] = inference_mses_imp[-1]

        # dendritic model
        pcn_den = RecPCN(dim=image_size ** 2, dendrite=True).to(device)
        optimizer = torch.optim.SGD(pcn_den.parameters(), lr=learning_lr)
        learning_mses_den = []
        for i in range(learning_iters):
            for batch_idx in range(0, sample_size, batch_size):
                data = X[batch_idx : batch_idx + batch_size]
                optimizer.zero_grad()
                pcn_den.learning(data)
                optimizer.step()
            mse = pcn_den.train_mse.cpu().detach().numpy()
            learning_mses_den.append(mse)

        inference_mses_den = []
        X_recon_den = X_c.clone()
        for j in range(inference_iters):
            delta_X = pcn_den.inference(X_recon_den)
            X_recon_den += inference_lr * (delta_X * update_mask)
            mse = torch.mean((X_recon_den - X) ** 2).cpu().detach().numpy()
            inference_mses_den.append(mse)

        dendritic_mses[c, s] = inference_mses_den[-1]

        # visualization
        fig, ax = plt.subplots(4, 2)
        ax[0, 0].plot(learning_mses_exp)
        ax[0, 0].set_title("exp learning")
        ax[0, 1].plot(inference_mses_exp)
        ax[0, 1].set_title("exp inference")
        ax[1, 0].plot(learning_mses_imp)
        ax[1, 0].set_title("imp learning")
        ax[1, 1].plot(inference_mses_imp)
        ax[1, 1].set_title("imp inference")
        ax[2, 0].plot(learning_mses_den)
        ax[2, 0].set_title("den learning")
        ax[2, 1].plot(inference_mses_den)
        ax[2, 1].set_title("den inference")
        # ax[3, 0].plot(learning_mses_den)
        ax[3, 0].set_title("hn learning")
        ax[3, 1].plot(inference_mses_hn)
        ax[3, 1].set_title("hn inference")
        plt.savefig(sub_path + "/MSEs")
        plt.show()
        plt.close()

        image_size = 5
        # minimum, maximum = torch.min(X).numpy(), torch.max(X).numpy()
        fig, ax = plt.subplots(5, 6, figsize=(10, 8))
        for i in range(5):
            ax[i, 0].imshow(
                X[i].cpu().detach().numpy().reshape(image_size, image_size),
                cmap="gray",
                vmin=-1,
                vmax=1,
            )
            ax[i, 0].axis("off")
            ax[i, 1].imshow(
                X_c[i].cpu().detach().numpy().reshape(image_size, image_size),
                cmap="gray",
                vmin=-1,
                vmax=1,
            )
            ax[i, 1].axis("off")
            ax[i, 2].imshow(
                X_recon_exp[i].cpu().detach().numpy().reshape(image_size, image_size),
                cmap="gray",
                vmin=-1,
                vmax=1,
            )
            ax[i, 2].axis("off")
            ax[i, 3].imshow(
                X_recon_imp[i].cpu().detach().numpy().reshape(image_size, image_size),
                cmap="gray",
                vmin=-1,
                vmax=1,
            )
            ax[i, 3].axis("off")
            ax[i, 4].imshow(
                X_recon_den[i].cpu().detach().numpy().reshape(image_size, image_size),
                cmap="gray",
                vmin=-1,
                vmax=1,
            )
            ax[i, 4].axis("off")
            ax[i, 5].imshow(
                X_recon_hn[i].cpu().detach().numpy().reshape(image_size, image_size),
                cmap="gray",
                vmin=-1,
                vmax=1,
            )
            ax[i, 5].axis("off")
        plt.savefig(sub_path + "/examples")
        plt.show()
        plt.close()

# final visualization
explicit_mean = np.mean(explicit_mses, axis=1)
explicit_std = np.std(explicit_mses, axis=1)
implicit_mean = np.mean(implicit_mses, axis=1)
implicit_std = np.std(implicit_mses, axis=1)
dendritic_mean = np.mean(dendritic_mses, axis=1)
dendritic_std = np.std(dendritic_mses, axis=1)
hn_mean = np.mean(hn_mses, axis=1)
hn_std = np.std(hn_mses, axis=1)

# plt.rcParams['figure.dpi'] = 150
plt.figure(figsize=(5, 3))
plt.errorbar(
    np.arange(len(cover_sizes)),
    explicit_mean,
    yerr=explicit_std,
    lw=2,
    label="explicit",
    color="#708A83",
)
plt.errorbar(
    np.arange(len(cover_sizes)),
    implicit_mean,
    yerr=implicit_std,
    lw=2,
    label="implicit",
    color="#B8BCBF",
)
plt.errorbar(
    np.arange(len(cover_sizes)),
    dendritic_mean,
    yerr=dendritic_std,
    lw=2,
    label="dendritic",
    color="#D77A41",
)
plt.errorbar(
    np.arange(len(cover_sizes)),
    hn_mean,
    yerr=hn_std,
    lw=2,
    label="Hopfield",
)

plt.legend(loc="best")
plt.xticks(np.arange(len(cover_sizes)), cover_sizes)
plt.xlabel("Number of masked pixels")
plt.ylabel("MSE")
plt.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.title("Retrieval MSE, different mask sizes")
plt.tight_layout()
plt.savefig(result_path + "/all_mses", dpi=400)
plt.show()
plt.close()
