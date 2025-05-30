import torch.nn as nn
import torch


class ActionNetwork(nn.Module):
    """
    The action network outputs the final classification.
    """
    def __init__(self, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Predict action/class from hidden state.
        
        Args:
            hidden_state: RNN hidden state (B, hidden_size)
            
        Returns:
            action_logits: class logits (B, num_classes)
        """
        return self.fc(hidden_state)
