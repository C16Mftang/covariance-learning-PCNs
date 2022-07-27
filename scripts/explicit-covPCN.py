import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from src.get_data import *
from src.models import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

learning_iters = 200
learning_lr = 1e-4
inference_iters = 200
inference_lr = 0.1
batch_size = 10
cover_size = 12
noise_var = 0.05
divisor = 2
image_size = 32
sample_size = 10
sample_size_test = 10

(X, _), (X_test, _) = get_cifar10('./data', 
                sample_size=sample_size, 
                sample_size_test=sample_size_test,
                batch_size=batch_size, 
                seed=10, 
                device=device,
                classes=None)
size = X.shape
X_c, update_mask = cover_bottom(X, divisor, device)
X = X.reshape(-1, size[-1]*size[-2]*size[-3])

pcn = ExplicitPCN(size[-1]*size[-2]*size[-3]).to(device)
optimizer = torch.optim.Adam(pcn.parameters(), lr=learning_lr)

# training
train_mses = []
for i in range(learning_iters):
    if i % 10 == 0:
        print(f'Epoch: {i}')
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        optimizer.zero_grad()
        pcn.learning(data)
        optimizer.step()
    train_mses.append(pcn.train_mse.cpu().detach().numpy())

# retrieval
X_recon = X_c.clone()
for j in range(inference_iters):
    delta_X = pcn.inference(X_recon)
    X_recon += inference_lr * (delta_X * update_mask)
mse_r = torch.mean((X_recon - X)**2, dim=1)

plt.figure()
plt.plot(train_mses)
plt.savefig('./results/explicit-mse')

fig, ax = plt.subplots(1, 3, figsize=(12, 5))
ax[0].imshow(X[0].reshape((size[1], image_size, image_size)).cpu().detach().numpy().transpose((1,2,0)))
ax[0].axis('off')
ax[1].imshow(X_c[0].reshape((size[1], image_size, image_size)).cpu().detach().numpy().transpose((1,2,0)))
ax[1].axis('off')
ax[2].imshow(X_recon[0].reshape((size[1], image_size, image_size)).cpu().detach().numpy().transpose((1,2,0)))
ax[2].axis('off')
plt.savefig('./results/explicit-imgs')