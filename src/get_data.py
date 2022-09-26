import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import random
import numpy as np

def cover_bottom(X, divisor, device):
    size = X.shape # 10, 3, 32, 32
    mask = torch.ones_like(X).to(device)
    mask[:, :, size[-2]//divisor+1:, :] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, :, size[-2]//divisor+1:, :] += 1
    update_mask = update_mask.reshape(-1, size[-1]*size[-2]*size[-3])
    X_c = (X * mask).to(device)
    X_c = X_c.reshape(-1, size[-1]*size[-2]*size[-3])

    return X_c, update_mask

def cover_center(X, cover_size, device):
    size = X.shape
    mask = torch.ones_like(X).to(device)
    mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] += 1
    update_mask = update_mask.reshape(-1, size[1]*size[2])
    X_c = (X * mask).to(device)
    X_c = X_c.reshape(-1, size[1]*size[2])

    return X_c, update_mask

def add_gaussian_noise(X, var, device):
    size = X.shape
    mask = (torch.randn(size) * np.sqrt(var)).to(device)
    update_mask = torch.ones_like(X).to(device)
    update_mask = update_mask.reshape(-1, size[-1]*size[-2]*size[-3])
    X_c = (X + mask).to(device)
    X_c = X_c.reshape(-1, size[-1]*size[-2]*size[-3])

    return X_c, update_mask


def get_2d_gaussian(sample_size, device, seed=10):
    dim = 2
    # simulation data
    mean = np.array([0,0])
    cov = np.array([[2,1],
                    [1,2]])
    np.random.seed(seed)
    X = np.random.multivariate_normal(mean, cov, size=sample_size)
    X_c = X * np.concatenate([np.ones((sample_size,1))]+[np.zeros((sample_size,1))]*(dim-1), axis=1)
    update_mask = np.concatenate([np.zeros((sample_size,1))]+[np.ones((sample_size,1))]*(dim-1), axis=1)

    X = torch.tensor(X).float().to(device)
    X_c = torch.tensor(X_c).float().to(device)
    update_mask = torch.tensor(update_mask).float().to(device)

    return X, X_c, update_mask


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

    return X


def get_mnist(datapath, sample_size, sample_size_test, batch_size, seed, device, classes=None):
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
    random.seed(seed)
    test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = [], []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
        y.append(targ)
    X = torch.cat(X, dim=0).to(device) # size, 28, 28
    y = torch.cat(y, dim=0).to(device)

    X_test, y_test = [], []
    for batch_idx, (data, targ) in enumerate(test_loader):
        X_test.append(data)
        y_test.append(targ)
    X_test = torch.cat(X_test, dim=0).to(device) # size, 28, 28
    y_test = torch.cat(y_test, dim=0).to(device)

    print(X.shape)
    return (X, y), (X_test, y_test)


def get_cifar10(datapath, sample_size, sample_size_test, batch_size, seed, device, classes=None):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(datapath, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(datapath, train=False, transform=transform, download=True)
    
     # subsetting data based on sample size and number of classes
    idx = sum(torch.tensor(train.targets) == c for c in classes).bool() if classes else range(len(train))
    train.targets = torch.tensor(train.targets)[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    random.seed(seed)
    test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = [], []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
        y.append(targ)
    X = torch.cat(X, dim=0).to(device) # size, c, 32, 32
    y = torch.cat(y, dim=0).to(device)

    X_test, y_test = [], []
    for batch_idx, (data, targ) in enumerate(test_loader):
        X_test.append(data)
        y_test.append(targ)
    X_test = torch.cat(X_test, dim=0).to(device) # size, 32, 32
    y_test = torch.cat(y_test, dim=0).to(device)

    return (X, y), (X_test, y_test)


def get_fashionMNIST(datapath, sample_size, batch_size, seed, device, classes=None):
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

    return X