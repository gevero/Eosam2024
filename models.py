import torch.nn as nn
from layers import  BaseBlock, LinearBlock, ResidualLayer

class MLP(nn.Module):
    def __init__(self, hidden_layers = [5,25,50,50,100,100,200,200]):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_layers) - 1):
            self.layers.append(BaseBlock(hidden_layers[i-1], hidden_layers[i]))
        self.final_block = LinearBlock(hidden_layers[-2], hidden_layers[-1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_block(x)
        return x
    
class ResidualMLP(nn.Module):
    def __init__(self, hidden_layers = [5,25,50,50,100,100,200,200]):
        super(ResidualMLP, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_layers) - 1):
            self.layers.append(ResidualLayer(hidden_layers[i-1], hidden_layers[i]))
        self.final_block = LinearBlock(hidden_layers[-2], hidden_layers[-1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_block(x)
        return x