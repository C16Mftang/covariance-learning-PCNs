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

# training parameters
sample_size = 500
inference_lr = 0.1
training_lr = 0.001
training_iters = 100
inference_iters = 100
batch_size = 1
fix_intact = True
dim = 2
artificial_data = True

# 2d data
X, X_c, update_mask = get_2d_gaussian(sample_size)
explicit_model = ExplicitPCN(dim).to(device)
optimizer = torch.optim.Adam(explicit_model.parameters(), lr=training_lr)

# training model params
for i in range(training_iters):
    for batch_idx in range(0, sample_size, batch_size):
        data = X[batch_idx:batch_idx+batch_size]
        optimizer.zero_grad()
        explicit_model.learning(data)
        optimizer.step()

# retrieval
X_recon = X_c.clone()
for j in range(inference_iters):
    delta_X = explicit_model.inference(X_recon)
    X_recon += inference_lr * (delta_X * update_mask)
mse = torch.mean((X_recon - X)**2, dim=1)

X = X.detach().cpu().numpy()
X_recon = X_recon.detach().cpu().numpy()
plt.figure()
plt.scatter(X[:,0], X[:,1], alpha=0.4, color='gray', label='Data')
plt.plot(X[:,0], X_recon[:,1], label='Explicit model', lw=3.5)
plt.show()
