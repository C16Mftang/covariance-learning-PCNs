import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from get_data import get_cifar10, get_fashionMNIST
from models import RecPCN

device = "cpu"


learning_iters = 100
# learning_lr = 0.0015
inference_iters = 100
inference_lr = 1.0
# sample_size = 10
batch_size = 10
cover_size = 12
image_size = 28

mses = []
for learning_lr, sample_size in zip([1e-3, 5e-4, 2e-4], [100,200,400]):
    X, X_c, update_mask = get_fashionMNIST('./data', 
                                        sample_size=sample_size, 
                                        batch_size=batch_size, 
                                        seed=10, 
                                        cover_size=cover_size, 
                                        device=device)
    pcn = RecPCN(dim=image_size**2).to(device)
    optimizer = torch.optim.SGD(pcn.parameters(), lr=learning_lr)

    for i in range(learning_iters):
        for batch_idx in range(0, sample_size, batch_size):
            data = X[batch_idx:batch_idx+batch_size]
            optimizer.zero_grad()
            pcn.learning(data)
            optimizer.step()

        X_recon = X_c.clone()
        for j in range(inference_iters):
            delta_X = pcn.inference(X_recon)
            X_recon += inference_lr * (delta_X * update_mask)

        # mse.append(torch.mean((X_recon - X)**2).cpu().detach().numpy())

    mses.append(torch.mean((X_recon - X)**2, dim=1))

plt.figure()
for mse, sample_size in zip(mses, [100,200,400]):    
    plt.hist(mse.cpu().detach().numpy(), density=True, bins='auto', alpha=0.4, edgecolor='k', label=f'{sample_size} memorized')
plt.legend()
plt.xlabel('MSE')
plt.ylabel('Density')
plt.show()

# plt.figure()
# plt.plot(mse)
# plt.show()

# fig, ax = plt.subplots(5, 3, figsize=(5,8))
# for i in range(5):
#     ax[i, 0].imshow(X[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
#     ax[i, 0].axis('off')
#     ax[i, 1].imshow(X_c[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
#     ax[i, 1].axis('off')
#     ax[i, 2].imshow(X_recon[i].cpu().detach().numpy().reshape(image_size, image_size), cmap='gray')
#     ax[i, 2].axis('off')
# plt.show()
