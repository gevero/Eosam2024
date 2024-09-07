import torch.nn as nn


class BaseBlock(nn.Module):
    def __init__(self, in_features, out_features, p=0.2, activation=nn.GELU()):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(p=p)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SigmaBlock(nn.Module):
    def __init__(self, in_features, out_features, p=0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=p)
        self.sigma = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigma(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


import torch.nn as nn


class ResidualLayer(nn.Module):
    """Fully connected residual layer with batch normalization, activation, and dropout.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        activation (str): Activation function name (e.g., 'relu', 'tanh').
        dropout_p (float): Dropout probability.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        act1 (nn.Module): First activation function.
        bn1 (nn.BatchNorm1d): Batch normalization layer after fc1.
        dropout1 (nn.Dropout): First dropout layer.
        fc2 (nn.Linear): Second fully connected layer.
        act2 (nn.Module): Second activation function.
        bn2 (nn.BatchNorm1d): Batch normalization layer after fc2.
        dropout2 (nn.Dropout): Second dropout layer.
    """

    def __init__(self, in_features, out_features, dropout_p=0.2):
        super().__init__()

        # first set of transformations
        self.fc1 = nn.Linear(in_features, out_features)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dropout1 = nn.Dropout(dropout_p)

        # If the input and output dimensions are different, add a projection layer
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = nn.Identity()

        # second set of linear transformations
        self.fc2 = nn.Linear(out_features, out_features)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x):
        """Forward pass through the residual layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # residual to be mapped later
        residual = self.projection(x)

        # first set of transformations
        x = self.fc1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # second set of transformations
        x = self.fc2(x)
        x += residual  # combine residual after second dense layer but before activation
        x = self.act2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        return x


class UpsampleConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        output_size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        self.upsample = nn.Upsample(size=output_size, mode="nearest")

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x
