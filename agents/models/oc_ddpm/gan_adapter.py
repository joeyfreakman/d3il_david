import hydra
import torch
from omegaconf import DictConfig
from torch import nn


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, encoder: DictConfig):
        super(Discriminator, self).__init__()
        self.encoder = hydra.utils.instantiate(encoder)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.model(x)
        return output


# Generator Model
class Generator(nn.Module):
    def __init__(self, encoder: DictConfig):
        super(Generator, self).__init__()
        self.encoder = hydra.utils.instantiate(encoder)

    def forward(self, z):
        output = self.model(z)
        output = output.view(output.size(0), 1, 28, 28)
        return output