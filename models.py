import torch.nn as nn
from layers import BaseBlock, LinearBlock, SigmaBlock


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # first layer
        self.linear1 = nn.Linear(5, 1000, bias=False)
        self.activation1 = nn.ReLU()

        # second layer
        self.linear2 = nn.Linear(1000, 1000, bias=False)
        self.activation2 = nn.ReLU()

        # third layer
        self.linear3 = nn.Linear(1000, 1000, bias=False)
        self.activation3 = nn.ReLU()

        # fourth layer
        self.linear4 = nn.Linear(1000, 1000, bias=False)
        self.activation4 = nn.ReLU()

        # fifth layer
        self.linear5 = nn.Linear(1000, 1000, bias=False)
        self.activation5 = nn.ReLU()

        # sixth layer
        self.linear6 = nn.Linear(1000, 200, bias=False)

    def forward(self, x):

        # forward pass of the nn
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.linear5(x)
        x = self.activation5(x)
        x = self.linear6(x)

        return x


class FlexibleMLP(nn.Module):
    def __init__(
        self,
        hidden_layers=[5, 1000, 1000, 1000, 1000, 1000, 200],
        p=0.2,
        activation=nn.GELU(),
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_layers) - 1):
            self.layers.append(
                BaseBlock(
                    hidden_layers[i - 1], hidden_layers[i], p=p, activation=activation
                )
            )
        self.final_block = LinearBlock(hidden_layers[-2], hidden_layers[-1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_block(x)
        return x


class FlexibleInverseMLP(nn.Module):
    def __init__(
        self,
        hidden_layers=[200, 1000, 1000, 1000, 1000, 1000, 2, 3, 3],
        p=0.2,
        activation=nn.GELU(),
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_layers) - 3):
            self.layers.append(
                BaseBlock(
                    hidden_layers[i - 1], hidden_layers[i], p=p, activation=activation
                )
            )

        # define 2 classification heads and 1 regression head
        self.lattice_head = LinearBlock(hidden_layers[-4], hidden_layers[-3])
        self.material_head = LinearBlock(hidden_layers[-4], hidden_layers[-2])
        self.geometry_head = LinearBlock(hidden_layers[-4], hidden_layers[-1])

    def forward(self, x):

        # common path
        for layer in self.layers:
            x = layer(x)

        # three heads
        x_lattice = self.lattice_head(x)
        x_material = self.material_head(x)
        x_geometry = self.geometry_head(x)

        return x_lattice,x_material,x_geometry
