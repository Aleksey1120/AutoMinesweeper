import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class MinesweeperModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, block_count):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        ))
        for _ in range(block_count):
            self.layers.append(ResidualBlock(hidden_channels, hidden_channels))
        self.layers.append(nn.Conv2d(hidden_channels, 1, (1, 1)))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return torch.squeeze(self.layers(x), dim=1)

    def get_parameters_count(self):
        return sum([p.numel() for p in self.parameters()])
