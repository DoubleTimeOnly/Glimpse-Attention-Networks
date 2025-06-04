import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class ConvBNReLU(nn.Module):
    """
    Helper module that bundles a 2D convolution, batch normalization, and ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ContextNetwork(nn.Module):
    def __init__(self, out_channels: int, input_wh: tuple[int, int]):
        super().__init__()
        self._conv1 = ConvBNReLU(3, 64, kernel_size=5, padding=2)
        self._conv2 = ConvBNReLU(64, 64, kernel_size=3, padding=1)
        self._conv3 = ConvBNReLU(64, 128, kernel_size=3, padding=1)

        self._fc = nn.Linear(128*input_wh[0]*input_wh[1], out_channels)
        self._input_shape = input_wh[::-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.resize(x, self._input_shape)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x
        