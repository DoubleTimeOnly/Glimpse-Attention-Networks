import torch
import torch.nn as nn
import torch.nn.functional as F


class GlimpseSensor(nn.Module):
    """
    The glimpse sensor extracts a retina-like representation around a given location.
    It extracts multiple resolution patches centered at the location.
    """
    def __init__(self, glimpse_size: int = 8, num_patches: int = 1):
        super().__init__()
        self.glimpse_size = glimpse_size
        self.num_patches = num_patches
        
    def forward(self, x: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        """
        Extract glimpse from image x at location.
        
        Args:
            x: input image tensor of shape (B, C, H, W)
            location: glimpse locations tensor of shape (B, 2) with values in [-1, 1]
            
        Returns:
            glimpse: tensor of shape (B, num_patches * C * glimpse_size^2)
        """
        batch_size, channels, height, width = x.shape
        
        # Convert location from [-1, 1] to pixel coordinates
        glimpse_patches = []
        
        for i in range(self.num_patches):
            # Calculate the size of this patch (each patch is 2x larger than previous)
            patch_size = self.glimpse_size * (2 ** i)
            
            # Extract patch using grid_sample
            # Create a grid for the patch
            theta = torch.zeros(batch_size, 2, 3, device=x.device)
            
            # Scale factor to extract patch of desired size
            scale = patch_size / min(height, width)
            theta[:, 0, 0] = scale
            theta[:, 1, 1] = scale
            
            # Translation to center the patch at the location
            theta[:, 0, 2] = location[:, 0]  # x coordinate
            theta[:, 1, 2] = location[:, 1]  # y coordinate
            
            # Create sampling grid
            grid = F.affine_grid(theta, [batch_size, channels, patch_size, patch_size], align_corners=False)
            
            # Sample the patch
            patch = F.grid_sample(x, grid, align_corners=False)
            
            # Resize patch to glimpse_size
            patch = F.interpolate(patch, size=(self.glimpse_size, self.glimpse_size), mode='bilinear', align_corners=False)
            
            glimpse_patches.append(patch.view(batch_size, -1))
        
        # Concatenate all patches
        glimpse = torch.cat(glimpse_patches, dim=1)
        return glimpse
