import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

    # run the following if this class inherits object.
    # def __call__(self, inp):
    #     return self.forward(inp)

class ReLU(nn.Module):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,))