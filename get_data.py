import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import random


def get_cifar10(datapath, sample_size, batch_size, seed, cover_size, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(datapath, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(datapath, train=False, transform=transform, download=True)
    
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
    mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] += 1
    update_mask = update_mask.reshape(-1, 32**2)

    X_c = (X * mask).to(device) # size, 32, 32

    X = X.reshape(-1, 32**2) # size, 1024
    X_c = X_c.reshape(-1, 32**2) # size, 1024

    return X, X_c, update_mask


def get_fashionMNIST(datapath, sample_size, batch_size, seed, cover_size, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.FashionMNIST(datapath, train=True, transform=transform, download=True)
    test = datasets.FashionMNIST(datapath, train=False, transform=transform, download=True)
    
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
    mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] += 1
    update_mask = update_mask.reshape(-1, 28**2)

    X_c = (X * mask).to(device) # size, 28, 28

    X = X.reshape(-1, 28**2) # size, 784
    X_c = X_c.reshape(-1, 28**2) # size, 784

    return X, X_c, update_mask