import torch
import torch.nn as nn
from layers import  LinearBlock, SigmaBlock

class MLP(nn.Module):
    def __init__(self, hidden_layers):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_layers) - 1):
            self.layers.append(LinearBlock(hidden_layers[i-1], hidden_layers[i]))
        self.sigma_block = SigmaBlock(hidden_layers[-2], hidden_layers[-1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.sigma_block(x)
        return x