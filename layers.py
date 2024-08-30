import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batchnorm = nn.BatchNorm1d(out_features)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.elu(x)
        return x
    

class SigmaBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(SigmaBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batchnorm = nn.BatchNorm1d(out_features)
        self.sigma = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.sigma(x)
        return x
    
    
class UpsampleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride=1, padding=0, dilation=1):
        super(UpsampleConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.upsample = nn.Upsample(size=output_size, mode='nearest')

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x