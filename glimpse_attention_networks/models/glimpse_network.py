import torch
import torch.nn as nn

from glimpse_attention_networks.models.glimpse_sensor import GlimpseSensor


class GlimpseNetwork(nn.Module):
    """
    The glimpse network combines the retina representation with the location information.
    """
    def __init__(
        self, 
        glimpse_size: int = 7, 
        num_patches: int = 1, 
        channels: int = 1,
        hidden_size: int = 127, 
        glimpse_hidden: int = 256
    ):
        super().__init__()
        
        self.sensor = GlimpseSensor(glimpse_size, num_patches)
        
        # Networks for processing glimpse and location
        glimpse_input_size = num_patches * channels * glimpse_size * glimpse_size
        self.glimpse_fc = nn.Sequential(
            nn.Linear(glimpse_input_size, hidden_size),
            nn.ReLU()
        )
        
        # Updated to handle 2D coordinates
        self.location_fc = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU()
        )
        
        # Combine glimpse and location information
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, glimpse_hidden),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        """
        Process glimpse and location to produce glimpse representation.
        
        Args:
            x: input image (B, C, H, W)
            location: glimpse location (B, 2) - (x, y) coordinates
            
        Returns:
            glimpse representation (B, glimpse_hidden)
        """
        # Extract glimpse
        glimpse = self.sensor(x, location)
        
        # Process glimpse and location separately
        glimpse_feat = self.glimpse_fc(glimpse)
        location_feat = self.location_fc(location)
        
        # Combine features
        combined = torch.cat([glimpse_feat, location_feat], dim=1)  # Changed dim=0 to dim=1
        glimpse_repr = self.combined_fc(combined)
        
        return glimpse_repr
