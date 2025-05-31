import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class ContextNetwork(nn.Module):
    def __init__(self, out_channels: int, input_wh: tuple[int, int]):
        super().__init__()
        self._conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self._conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self._conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

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
        