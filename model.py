import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, inner_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(inner_channels, 1, (1, 1))
        )

    def forward(self, x):
        return torch.squeeze(self.layers(x), dim=1)

    def get_parameters_count(self):
        return sum([p.numel() for p in self.parameters()])
