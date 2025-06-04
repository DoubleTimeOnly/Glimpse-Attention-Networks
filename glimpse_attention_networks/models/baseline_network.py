import torch.nn as nn
import torch

class BaselineNetwork(nn.Module):
    """
    Baseline network for variance reduction in REINFORCE.
    """
    def __init__(self, hidden_size: int = 256):
        super(BaselineNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Predict baseline value.
        
        Args:
            hidden_state: RNN hidden state (B, hidden_size)
            
        Returns:
            baseline: predicted value (B, 1)
        """
        return self.fc(hidden_state)
