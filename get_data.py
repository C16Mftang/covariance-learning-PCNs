import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import random

def create_gaussian(savepath, seed=10):
    # create random covariance matrix
    torch.manual_seed(seed)
    G = torch.randn(25, 25)
    S = torch.matmul(G, G.t()) / 10.
    L = torch.linalg.cholesky(S)

    # sample from this normal
    m = MultivariateNormal(torch.zeros(25), scale_tril=L)
    X = m.sample(sample_shape=(5000,))
    X = X.reshape(-1, 5, 5)
    torch.save(X, savepath+'/gaussian_data.pt')


def get_gaussian(datapath, sample_size, batch_size, seed, device):
    train = torch.load(datapath+'/gaussian_data.pt')
    if sample_size != len(train):
        random.seed(seed)
        train = train[random.sample(range(len(train)), sample_size)] # size, 5, 5
    
    X = torch.tensor(train)
    # create the corrupted version
    size = X.shape
    mask = torch.ones_like(X).to(device)
    # mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    mask[:, size[1]//2+1:, :] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, size[1]//2+1:, :] += 1
    update_mask = update_mask.reshape(-1, size[1]*size[2])

    X_c = (X * mask).to(device) # size, 5, 5

    X = X.reshape(-1, size[1]*size[2]) # size, 25
    X_c = X_c.reshape(-1, size[1]*size[2]) # size, 25

    return X, X_c, update_mask


def get_mnist(datapath, sample_size, batch_size, seed, cover_size, device, classes=None):
    # classes: a list of specific class to sample from
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
    test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

    # subsetting data based on sample size and number of classes
    idx = sum(train.targets == c for c in classes).bool() if classes else range(len(train))
    train.targets = train.targets[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X = []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
    X = torch.cat(X, dim=0).squeeze().to(device) # size, 28, 28

    # create the corrupted version
    size = X.shape
    mask = torch.ones_like(X).to(device)
    # mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    mask[:, size[1]//2:, :] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, size[1]//2:, :] += 1
    update_mask = update_mask.reshape(-1, 28**2)

    X_c = (X * mask).to(device) # size, 28, 28

    X = X.reshape(-1, 28**2) # size, 784
    X_c = X_c.reshape(-1, 28**2) # size, 784

    return X, X_c, update_mask


def get_cifar10(datapath, sample_size, batch_size, seed, cover_size, device, classes=None):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(datapath, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(datapath, train=False, transform=transform, download=True)
    
     # subsetting data based on sample size and number of classes
    idx = sum(train.targets == c for c in classes).bool() if classes else range(len(train))
    train.targets = torch.tensor(train.targets)[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X = []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
    X = torch.cat(X, dim=0).squeeze().to(device) # size, 32, 32

    # create the corrupted version
    size = X.shape
    mask = torch.ones_like(X).to(device)
    mask[:, size[1]//2:, :] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, size[1]//2:, :] += 1
    update_mask = update_mask.reshape(-1, 32**2)

    X_c = (X * mask).to(device) # size, 32, 32

    X = X.reshape(-1, 32**2) # size, 1024
    X_c = X_c.reshape(-1, 32**2) # size, 1024

    return X, X_c, update_mask


def get_fashionMNIST(datapath, sample_size, batch_size, seed, cover_size, device, classes=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.FashionMNIST(datapath, train=True, transform=transform, download=True)
    test = datasets.FashionMNIST(datapath, train=False, transform=transform, download=True)
    
    # subsetting data based on sample size and number of classes
    idx = sum(train.targets == c for c in classes).bool() if classes else range(len(train))
    train.targets = train.targets[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X = []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
    X = torch.cat(X, dim=0).squeeze().to(device) # size, 28, 28

    # create the corrupted version
    size = X.shape
    mask = torch.ones_like(X).to(device)
    # mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    mask[:, size[1]//2:, :] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, size[1]//2:, :] += 1
    update_mask = update_mask.reshape(-1, 28**2)

    X_c = (X * mask).to(device) # size, 28, 28

    X = X.reshape(-1, 28**2) # size, 784
    X_c = X_c.reshape(-1, 28**2) # size, 784

    return X, X_c, update_mask