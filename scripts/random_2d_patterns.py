import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from src.get_data import *
from src.models import *
from src.utils import cov

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# training parameters
sample_size = 500
inference_lr = 0.1 # could be more fine-tuned for each model
training_lr = 0.0001 # could be more fine-tuned for each model
training_iters = 100
inference_iters = 100
batch_size = 1
dim = 2

X, X_c, update_mask = get_2d_gaussian(sample_size, device)
explicit_model = ExplicitPCN(dim).to(device)
optimizer_ex = torch.optim.SGD(explicit_model.parameters(), lr=training_lr)
implicit_model = RecPCN(dim, dendrite=False).to(device)
optimizer_im = torch.optim.SGD(implicit_model.parameters(), lr=training_lr)
dendritic_model = RecPCN(dim, dendrite=True).to(device)
optimizer_den = torch.optim.SGD(dendritic_model.parameters(), lr=training_lr)
# sample covariance matrix
ML_cov = cov(X)

# training model params
def memory_retrieval(model, optimizer):
    for i in range(training_iters):
        for batch_idx in range(0, sample_size, batch_size):
            data = X[batch_idx:batch_idx+batch_size]
            optimizer.zero_grad()
            model.learning(data)
            optimizer.step()

    # retrieval
    X_recon = X_c.clone()
    for j in range(inference_iters):
        delta_X = model.inference(X_recon)
        X_recon += inference_lr * (delta_X * update_mask)
    mse = torch.mean((X_recon - X)**2, dim=1)

    return X_recon

X_recon_ex = memory_retrieval(explicit_model, optimizer_ex).detach().cpu().numpy()
X_recon_im = memory_retrieval(implicit_model, optimizer_im).detach().cpu().numpy()
X_recon_den = memory_retrieval(dendritic_model, optimizer_den).detach().cpu().numpy()

X = X.detach().cpu().numpy()
# plot the learned weight matrices
ML_cov = ML_cov.cpu().detach().numpy()

# plot the learned linear retrievals
plt.figure()
plt.scatter(X[:,0], X[:,1], alpha=0.4, color='gray', label='Data')
plt.plot(X[:,0], X_recon_ex[:,1], label='Explicit model', lw=3.5)
plt.plot(X[:,0], X_recon_im[:,1], label='Implicit model', lw=3)
plt.plot(X[:,0], X_recon_den[:,1], label='Dendritic model', lw=2.5)
plt.plot(X[:,0], (ML_cov[0,1]/ML_cov[1,1])*X[:,0], label=r'$\Sigma_{21}/\Sigma_{11}$')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
# ax.set_title(r'Linear prediction of $x_2$ from $x_1$')
# plt.savefig('./figs/gaussian2', dpi=400)
plt.show()

W_ex = explicit_model.S.cpu().detach().numpy()
W_im = implicit_model.Wr.cpu().detach().numpy()

print(W_ex, W_im)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(ML_cov, vmin=0, vmax=1)
ax[1].imshow(W_ex, vmin=0, vmax=1)
im = ax[2].imshow(W_im, vmin=0, vmax=1)
for i in range(3):
    ax[i].axis('off') 
fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
# plt.savefig('./figs/covmats_gaussian2', dpi=400)
plt.show()
