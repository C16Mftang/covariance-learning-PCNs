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

learning_iters = 400
learning_lr = 1e-4
inference_iters = 5000
inference_lr_hy = 1.
inference_lr_r = 0.5
batch_size = 200
cover_size = 12
noise_var = 0.05
divisor = 2
image_size = 28
nodes = [image_size**2//2, image_size**2//2, image_size**2]
sample_size = 10000
sample_size_test = 10

X, X_test = get_mnist('./data', 
                sample_size=sample_size, 
                sample_size_test=sample_size_test,
                batch_size=batch_size, 
                seed=10, 
                device=device,
                classes=[5,6])
size = X.shape
X_c, update_mask = cover_bottom(X, divisor, device)
X = X.reshape(-1, size[-1]*size[-2]*size[-3])

pcn_r = RecPCN(size[-1]*size[-2]*size[-3], dendrite=True).to(device)
optimizer_r = torch.optim.Adam(pcn_r.parameters(), lr=learning_lr)

########################################################################
# recurrent PCN
train_mses_r = []
for i in range(learning_iters):
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        optimizer_r.zero_grad()
        pcn_r.learning(data)
        optimizer_r.step()
    train_mses_r.append(pcn_r.train_mse.cpu().detach().numpy())

X_recon_r = X_c.clone()
for j in range(inference_iters):
    delta_X = pcn_r.inference(X_recon_r)
    X_recon_r += inference_lr_r * (delta_X * update_mask)
mse_r = torch.mean((X_recon_r - X)**2, dim=1)

plt.figure()
plt.plot(train_mses_r)
plt.show()

fig, ax = plt.subplots(3, 20, figsize=(24, 4))
for i in range(20):
    ax[0, i].imshow(X[i].reshape((size[1], image_size, image_size)).cpu().detach().numpy().transpose((1,2,0)), cmap='gray')
    ax[0, i].axis('off')
    ax[1, i].imshow(X_c[i].reshape((size[1], image_size, image_size)).cpu().detach().numpy().transpose((1,2,0)), cmap='gray')
    ax[1, i].axis('off')
    ax[2, i].imshow(X_recon_r[i].reshape((size[1], image_size, image_size)).cpu().detach().numpy().transpose((1,2,0)), cmap='gray')
    ax[2, i].axis('off')
plt.show()

