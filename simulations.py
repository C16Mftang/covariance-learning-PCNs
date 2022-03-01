import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from get_data import *
from models import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Friston PCN
learning_iters = 100
learning_lrs = [0.005, 0.005, 0.01]
inference_iters = 100
inference_lrs = [0.1, 0.1, 0.1]
batch_size = 10
cover_size = 12
image_size = 5
sample_size = 20

X = get_gaussian('./data', 
                sample_size=sample_size, 
                batch_size=batch_size, 
                seed=1,  
                device=device)

X_c, update_mask = cover_bottom_half(X, device)
X = X.reshape(-1, image_size*image_size)

################################################
# covariance model
pcn_F = FristonPCN(dim=image_size**2).to(device)
optimizer = torch.optim.SGD(pcn_F.parameters(), lr=learning_lrs[0])
for i in range(learning_iters):
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        optimizer.zero_grad()
        pcn_F.learning(data)
        optimizer.step()

X_recon_F = X_c.clone()
for j in range(inference_iters):
    delta_X = pcn_F.inference(X_recon_F)
    X_recon_F += inference_lrs[0] * (delta_X * update_mask)

mse_F = torch.mean((X_recon_F - X)**2, dim=1)

################################################
# recurrent model (non dendritic)
pcn_r = RecPCN(dim=image_size**2, dendrite=False)
optimizer = torch.optim.SGD(pcn_r.parameters(), lr=learning_lrs[1])
for i in range(learning_iters):
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        optimizer.zero_grad()
        pcn_r.learning(data)
        optimizer.step()

X_recon_r = X_c.clone()
for j in range(inference_iters):
    delta_X = pcn_r.inference(X_recon_r)
    X_recon_r += inference_lrs[1] * (delta_X * update_mask)

mse_r = torch.mean((X_recon_r - X)**2, dim=1)

################################################
# dendritic model
pcn_d = RecPCN(dim=image_size**2, dendrite=True)
optimizer = torch.optim.SGD(pcn_d.parameters(), lr=learning_lrs[2])
for i in range(learning_iters):
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        optimizer.zero_grad()
        pcn_d.learning(data)
        optimizer.step()

X_recon_d = X_c.clone()
for j in range(inference_iters):
    delta_X = pcn_d.inference(X_recon_d)
    X_recon_d += inference_lrs[2] * (delta_X * update_mask)

mse_d = torch.mean((X_recon_d - X)**2, dim=1)

image_size = 5
minimum, maximum = torch.min(X).numpy(), torch.max(X).numpy()
fig, ax = plt.subplots(5, 5, figsize=(8,8))
for i in range(5):
    ax[i, 0].imshow(X[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray', vmin=-1, vmax=1)
    ax[i, 0].axis('off')
    ax[i, 1].imshow(X_c[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray', vmin=-1, vmax=1)
    ax[i, 1].axis('off')
    ax[i, 2].imshow(X_recon_F[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray', vmin=-1, vmax=1)
    ax[i, 2].axis('off')
    ax[i, 3].imshow(X_recon_r[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray', vmin=-1, vmax=1)
    ax[i, 3].axis('off')
    ax[i, 4].imshow(X_recon_d[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray', vmin=-1, vmax=1)
    ax[i, 4].axis('off')
plt.savefig('./figs/gaussian5x5_all', dpi=400)
plt.show()
