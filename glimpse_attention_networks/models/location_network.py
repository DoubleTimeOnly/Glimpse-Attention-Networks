import torch
import torch.nn as nn


class LocationNetwork(nn.Module):
    """
    The location network outputs the mean of the next location to attend to.
    """
    def __init__(self, hidden_size: int = 256, std: float = 0.17):
        super(LocationNetwork, self).__init__()
        self.fc = nn.Linear(hidden_size, 2)
        self.std = std
        
    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next location mean and sample location.
        
        Args:
            hidden_state: RNN hidden state (B, hidden_size)
            
        Returns:
            location_mean: mean of location distribution (B, 2)
            location_sample: sampled location (B, 2)
        """
        location_mean = torch.tanh(self.fc(hidden_state))
        
        # Sample location from normal distribution
        if self.training:
            location_sample = location_mean + self.std * torch.randn_like(location_mean)
            location_sample = torch.tanh(location_sample)  # Bound to [-1, 1]
        else:
            location_sample = location_mean
            
        return location_mean, location_sample
