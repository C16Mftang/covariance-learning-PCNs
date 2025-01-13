import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import src.utils as utils

class ExplicitPCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # initialize parameters
        self.S = nn.Parameter(torch.eye(dim))
        self.mu = nn.Parameter(torch.zeros((dim,)))

    def forward(self, X):
        return X

    def learning(self, X):
        errs = torch.matmul(self.forward(X) - self.mu, torch.linalg.inv(self.S).T)
        grad_S = 0.5 * (torch.matmul(errs.T, errs) / X.shape[0] - torch.linalg.inv(self.S))
        grad_mu = torch.sum(errs, dim=0)

        self.S.grad = -grad_S
        self.mu.grad = -grad_mu
        self.train_mse = torch.mean(errs**2)

    def inference(self, X_c):
        errs_X = torch.matmul(self.forward(X_c) - self.mu, torch.linalg.inv(self.S).T)
        delta_X = -errs_X

        return delta_X

class RecPCN(nn.Module):
    def __init__(self, dim, dendrite=True, mode='linear'):
        super().__init__()
        self.dim = dim
        self.dendrite = dendrite
        # initialize parameters
        self.Wr = nn.Parameter(torch.zeros((dim, dim)))
        self.mu = nn.Parameter(torch.zeros((dim,)))

        if mode == 'linear':
            self.nonlin = utils.Linear()
        elif mode == 'rate':
            self.nonlin = utils.Sigmoid()
        elif mode == 'binary':
            self.nonlin = utils.Binary()
        elif mode == 'tanh':
            self.nonlin = utils.Tanh()
        else:
            raise ValueError("no such nonlinearity!")

    def forward(self, X):
        preds = torch.matmul(self.nonlin(X), self.Wr.t()) + self.mu
        return preds
        
    def learning(self, X):
        preds = self.forward(X)
        errs = X - preds
        grad_Wr = torch.matmul(errs.T, self.nonlin(X)).fill_diagonal_(0)
        grad_mu = torch.sum(errs, dim=0)

        self.Wr.grad = -grad_Wr
        self.mu.grad = -grad_mu
        self.train_mse = torch.mean(errs**2)

    def inference(self, X_c):
        errs_X = X_c - self.forward(X_c)
        delta_X = -errs_X if self.dendrite else (-errs_X + self.nonlin.deriv(X_c) * torch.matmul(errs_X, self.Wr))

        return delta_X

    def energy(self, X):
        """Calculate the energy value of the implicit model given a batch of input
        Aka the squared error

        Returns a vector of energies, indicating the energy value for each input in X
        """
        return torch.linalg.norm((X - self.forward(X)), dim=1)

class HopfieldNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.zeros((dim, dim)))

    def forward(self, X):
        return torch.sign(torch.matmul(X, self.W))

    def learning(self, patterns):
        num_patterns = patterns.shape[0]
        self.W.data = torch.matmul(patterns.T, patterns) / num_patterns
        # self.W.data.fill_diagonal_(0)

    def inference(self, X, steps=10):
        for _ in range(steps):
            X = self.forward(X)
        return X

class MultilayerPCN(nn.Module):
    def __init__(self, nodes, nonlin, Dt, lamb=0, use_bias=False):
        super().__init__()
        self.n_layers = len(nodes)
        self.layers = nn.Sequential()
        for l in range(self.n_layers-1):
            self.layers.add_module(f'layer_{l}', nn.Linear(
                in_features=nodes[l],
                out_features=nodes[l+1],
                bias=use_bias,
            ))

        self.mem_dim = nodes[0]
        self.memory = nn.Parameter(torch.zeros((nodes[0],)))
        self.Dt = Dt

        if nonlin == 'Tanh':
            nonlin = utils.Tanh()
        elif nonlin == 'ReLU':
            nonlin = utils.ReLU()
        elif nonlin == 'Linear':
            nonlin = utils.Linear()
        self.nonlins = [nonlin] * (self.n_layers - 1)
        self.use_bias = use_bias
        self.lamb = lamb

    def initialize(self):
        self.val_nodes = [[] for _ in range(self.n_layers)]
        self.preds = [[] for _ in range(self.n_layers)]
        self.errs = [[] for _ in range(self.n_layers)]

    def forward(self):
        val = self.memory.clone().detach()
        for l in range(self.n_layers-1):
            val = self.layers[l](self.nonlins[l](val))
        return val

    def update_err_nodes(self):
        raise NotImplementedError()

    def set_nodes(self, batch_inp):
        # computing val nodes
        self.val_nodes[0] = self.memory.clone().detach()
        for l in range(1, self.n_layers-1):
            self.val_nodes[l] = self.layers[l-1](self.nonlins[l-1](self.val_nodes[l-1]))
        self.val_nodes[-1] = batch_inp.clone()

        # computing error nodes
        self.update_err_nodes()

    def update_grads(self):
        raise NotImplementedError()

    def train_pc_generative(self, batch_inp, n_iters, update_mask):
        self.initialize()
        self.set_nodes(batch_inp)
        for itr in range(n_iters):
            self.update_val_nodes(update_mask)
        self.update_grads()
        self.register_buffer('top_activities', self.val_nodes[0])

    def test_pc_generative(self, corrupt_inp, n_iters, update_mask, sensory=True):
        self.initialize()
        self.set_nodes(corrupt_inp)
        for itr in range(n_iters):
            self.update_val_nodes(update_mask, recon=True)

        if sensory:
            return self.val_nodes[-1]
        else:
            return self.preds[-1]

    def energy(self):
        """Function to obtain the sum of all layers' squared MSE"""
        total_energy = 0
        for l in range(self.n_layers):
            total_energy += torch.sum(self.errs[l] ** 2)
        return total_energy
    
    
class HierarchicalPCN(MultilayerPCN):
    def __init__(self, nodes, nonlin, Dt, init_std=0., lamb=0, use_bias=False):
        super().__init__(nodes, nonlin, Dt, lamb, use_bias)
        self.memory = nn.Parameter(init_std * torch.randn((self.mem_dim,)))

    def update_err_nodes(self):
        for l in range(0, self.n_layers):
            if l == 0:
                self.preds[l] = self.memory.clone().detach()
            else:
                self.preds[l] = self.layers[l-1](self.nonlins[l-1](self.val_nodes[l-1]))
            self.errs[l] = self.val_nodes[l] - self.preds[l]
    
    def update_val_nodes(self, update_mask, recon=False):
        with torch.no_grad():
            for l in range(0, self.n_layers-1):
                derivative = self.nonlins[l].deriv(self.val_nodes[l])
                penalty = self.lamb if l == 0 else 0.
                delta = -self.errs[l] - penalty * torch.sign(self.val_nodes[l]) + derivative * torch.matmul(self.errs[l+1], self.layers[l].weight)
                self.val_nodes[l] = self.val_nodes[l] + self.Dt * delta
            if recon:
                # relax sensory layer value nodes if its corrupted (during reconstruction phase)
                self.val_nodes[-1] = self.val_nodes[-1] + self.Dt * (-self.errs[-1] * update_mask)

            self.update_err_nodes()

    def update_grads(self):
        self.memory.grad = -torch.sum(self.errs[0], axis=0)
        for l in range(self.n_layers-1):
            grad_w = -torch.matmul(self.errs[l+1].t(), self.nonlins[l](self.val_nodes[l]))
            self.layers[l].weight.grad = grad_w
            if self.use_bias:
                self.layers[l].bias.grad = -torch.sum(self.errs[l+1], axis=0)


class HybridPCN(MultilayerPCN):
    def __init__(self, nodes, nonlin, Dt, dendritic=True, init_std=0., init_std_Wr=0., lamb=0, use_bias=False):
        super().__init__(nodes, nonlin, Dt, lamb, use_bias)
        self.memory = nn.Parameter(init_std * torch.randn((self.mem_dim,)))
        self.Wr = nn.Parameter(init_std_Wr * torch.randn((self.mem_dim, self.mem_dim)))
        self.dendritic = dendritic

    def update_err_nodes(self):
        for l in range(0, self.n_layers):
            if l == 0:
                self.preds[l] = self.memory.clone().detach() + torch.matmul(self.val_nodes[l], self.Wr.t())
            else:
                self.preds[l] = self.layers[l-1](self.nonlins[l-1](self.val_nodes[l-1]))
            self.errs[l] = self.val_nodes[l] - self.preds[l]

    def update_val_nodes(self, update_mask, recon=False):
        with torch.no_grad():
            for l in range(0, self.n_layers-1):
                derivative = self.nonlins[l].deriv(self.val_nodes[l])
                penalty = self.lamb if l == 0 else 0.
                delta = -self.errs[l] - penalty * torch.sign(self.val_nodes[l]) + derivative * torch.matmul(self.errs[l+1], self.layers[l].weight)
                if l == 0 and self.dendritic is False:
                    delta = delta + torch.matmul(self.errs[l], self.Wr)
                self.val_nodes[l] = self.val_nodes[l] + self.Dt * delta
            if recon:
                # relax sensory layer value nodes if its corrupted (during reconstruction phase)
                self.val_nodes[-1] = self.val_nodes[-1] + self.Dt * (-self.errs[-1] * update_mask)

            self.update_err_nodes()

    def update_grads(self):
        self.memory.grad = -torch.sum(self.errs[0], axis=0)
        self.Wr.grad = -torch.matmul(self.errs[0].t(), self.val_nodes[0]).fill_diagonal_(0)

        for l in range(self.n_layers-1):
            grad_w = -torch.matmul(self.errs[l+1].t(), self.nonlins[l](self.val_nodes[l]))
            self.layers[l].weight.grad = grad_w
            if self.use_bias:
                self.layers[l].bias.grad = -torch.sum(self.errs[l+1], axis=0)


class DGPCN(nn.Module):
    def __init__(self, nodes, nonlin, Dt, lamb, MF_sparsity, Q_range=None):
        """Input variables:

        MF_sparsity: sparsity of the mossy fibre pathway (DG-CA3)
        """
        super().__init__()
        self.n_layers = len(nodes)

        # weight connecting EC to DG
        self.P = nn.Linear(nodes[0], nodes[1], bias=False)
        # nn.init.uniform_(self.P.weight, a=-1.0, b=1.0)

        # weight connecting DG to CA3, sparse
        self.Q = nn.Linear(nodes[1], nodes[2], bias=False)
        if Q_range:
            nn.init.uniform_(self.Q.weight, a=-Q_range, b=Q_range)
        # update mask of Q: randomly select MF weights that are plastic
        Q_mask = torch.where(torch.rand_like(self.Q.weight) < MF_sparsity, 0.0, 1.0)
        self.register_buffer("Q_mask", Q_mask)
        self.Q.weight = nn.Parameter(self.Q.weight.clone().detach() * self.Q_mask)

        # weight connecting CA3 to EC
        self.V = nn.Linear(nodes[2], nodes[0], bias=False)

        # recurrent weight in DG, initialized to be 0
        self.W_DG = nn.Linear(nodes[1], nodes[1], bias=False)
        nn.init.zeros_(self.W_DG.weight)

        # recurrent weight in CA3, initialized to be 0
        self.W_CA3 = nn.Linear(nodes[2], nodes[2], bias=True)
        nn.init.zeros_(self.W_CA3.weight)
        # nn.init.zeros_(self.W_CA3.bias)
        nn.init.normal_(self.W_CA3.bias, mean=0.0, std=1.0)
        
        self.Dt = Dt
        if nonlin == 'Tanh':
            self.nonlin = utils.Tanh()
        elif nonlin == 'ReLU':
            self.nonlin = utils.ReLU()
        self.lamb = lamb
        self.n_layers = len(nodes)
        assert self.n_layers == 3

    def initialize(self):
        self.val_nodes = [[] for _ in range(self.n_layers)]
        self.preds = [[] for _ in range(self.n_layers)]
        self.errs = [[] for _ in range(self.n_layers)]

    def forward(self, inp):
        val = self.P(self.nonlin(inp))
        val = self.Q(self.nonlin(val))
        val = self.V(self.nonlin(val))
        
        return val

    def update_err_nodes(self):
        # EC
        self.preds[0] = self.V(self.nonlin(self.val_nodes[2]))
        self.errs[0] = self.val_nodes[0] - self.preds[0]
        # DG
        # Nonlinear recurrent layer
        self.preds[1] = self.P(self.nonlin(self.val_nodes[0])) + self.W_DG(self.nonlin(self.val_nodes[1]))
        self.errs[1] = self.val_nodes[1] - self.preds[1]
        # CA3
        # Nonlinear recurrent layer
        self.preds[2] = self.Q(self.nonlin(self.val_nodes[1])) + self.W_CA3(self.nonlin(self.val_nodes[2]))
        self.errs[2] = self.val_nodes[2] - self.preds[2]

    def set_nodes(self, batch_inp):
        # EC
        self.val_nodes[0] = batch_inp.clone()
        # DG
        self.val_nodes[1] = self.P(self.nonlin(self.val_nodes[0]))
        # CA3
        self.val_nodes[2] = self.Q(self.nonlin(self.val_nodes[1])) + self.W_CA3.bias.detach().clone()

        # computing error nodes
        self.update_err_nodes()

    def update_val_nodes(self, update_mask, recon=False):
        with torch.no_grad():
            # EC
            if recon:
                # relax sensory layer value nodes if its corrupted (during reconstruction phase)
                self.val_nodes[0] = self.val_nodes[0] + self.Dt * (-self.errs[0] * update_mask)
                # self.lamb = 0.

            # DG
            # sparse penalty on activities; could also be on W_DG
            deriv_DG = self.nonlin.deriv(self.val_nodes[1])
            delta_DG = -self.errs[1] - self.lamb * torch.sign(self.val_nodes[1]) + deriv_DG * torch.matmul(self.errs[1], self.W_DG.weight)
            self.val_nodes[1] = self.val_nodes[1] + self.Dt * delta_DG

            # CA3
            deriv_CA3 = self.nonlin.deriv(self.val_nodes[2])
            delta_CA3 = -self.errs[2] + deriv_CA3 * torch.matmul(self.errs[2], self.W_CA3.weight) + deriv_CA3 * torch.matmul(self.errs[0], self.V.weight)
            self.val_nodes[2] = self.val_nodes[2] + self.Dt * delta_CA3

            self.update_err_nodes()

    def update_grads(self):
        self.P.weight.grad = -torch.matmul(self.errs[1].t(), self.nonlin(self.val_nodes[0]))
        self.W_DG.weight.grad = -torch.matmul(self.errs[1].t(), self.nonlin(self.val_nodes[1])).fill_diagonal_(0)

        Q_grad = -torch.matmul(self.errs[2].t(), self.nonlin(self.val_nodes[1]))
        self.Q.weight.grad = Q_grad * self.Q_mask
        self.W_CA3.weight.grad = -torch.matmul(self.errs[2].t(), self.nonlin(self.val_nodes[2])).fill_diagonal_(0)
        self.W_CA3.bias.grad = -torch.sum(self.errs[2], axis=0)

        self.V.weight.grad = -torch.matmul(self.errs[0].t(), self.nonlin(self.val_nodes[2]))

    def train_pc_generative(self, batch_inp, n_iters, update_mask):
        self.initialize()
        self.set_nodes(batch_inp)
        for itr in range(n_iters):
            self.update_val_nodes(update_mask)
        self.update_grads()


class AutoEncoder(nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        # initialize parameters
        self.We = nn.Linear(dim, latent_dim, bias=False)
        self.Wd = nn.Linear(latent_dim, dim, bias=False)

        self.tanh = nn.Tanh()
        
    def forward(self, X):
        return self.tanh(self.Wd(self.tanh(self.We(X))))
    