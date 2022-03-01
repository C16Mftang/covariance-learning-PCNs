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

learning_iters = 100
learning_lr = 1e-3
inference_iters = 100
inference_lr_hy = 1.
inference_lr_r = 1.
batch_size = 10
cover_size = 12
noise_var = 0.01
image_size = 28
nodes = [image_size**2//2, image_size**2//2, image_size**2]
sample_size = 100
sample_size_test = 10

X, X_test = get_mnist('./data', 
                sample_size=sample_size, 
                sample_size_test=sample_size_test,
                batch_size=batch_size, 
                seed=10, 
                device=device,
                classes=None)

X_c, update_mask = cover_bottom_half(X, device)
X = X.reshape(-1, image_size**2)
X_test = X_test.reshape(-1, image_size**2)

pcn_r = RecPCN(image_size**2, dendrite=True).to(device)
pcn_hy = HybridPCN(nodes, 'Tanh', inference_lr_hy, update_mask).to(device)
optimizer_r = torch.optim.Adam(pcn_r.parameters(), lr=learning_lr)
optimizer_hy = torch.optim.Adam(pcn_hy.parameters(), lr=learning_lr)

########################################################################
# recurrent PCN
for i in range(learning_iters):
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        optimizer_r.zero_grad()
        pcn_r.learning(data)
        optimizer_r.step()

X_recon_r = X_c.clone()
for j in range(inference_iters):
    delta_X = pcn_r.inference(X_recon_r)
    X_recon_r += inference_lr_r * (delta_X * update_mask)
mse_r = torch.mean((X_recon_r - X)**2, dim=1)

########################################################################
# hybrid PCN
for i in range(learning_iters):
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        optimizer_hy.zero_grad()
        pcn_hy.train_pc_generative(data, inference_iters)
        optimizer_hy.step()

X_recon_h = pcn_hy.test_pc_generative(X_c, inference_iters)
mse_h = torch.mean((X_recon_h - X)**2, dim=1)

fig, ax = plt.subplots(5, 4, figsize=(5,8))
for i in range(5):
    ax[i, 0].imshow(X[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
    ax[i, 0].axis('off')
    ax[i, 1].imshow(X_c[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
    ax[i, 1].axis('off')
    ax[i, 2].imshow(X_recon_r[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
    ax[i, 2].axis('off')
    ax[i, 3].imshow(X_recon_h[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
    ax[i, 3].axis('off')
plt.show()

plt.figure() 
plt.hist(mse_h.cpu().detach().numpy(), 
         density=True, 
         bins='auto', 
         alpha=0.4, 
         edgecolor='k', 
         label='Hybrid PCN')
plt.hist(mse_r.cpu().detach().numpy(), 
         density=True, 
         bins='auto', 
         alpha=0.4, 
         edgecolor='k', 
         label='Recurrent PCN')
plt.legend()
plt.xlabel('MSE')
plt.ylabel('Density')
# plt.title(f'Distribution of mse between {sample_size} retrieved and original MNIST test set')
# plt.savefig('/content/drive/MyDrive/Colab Notebooks/phd/figs/mse_dist_r_vs_h', dpi=300)
plt.show()

