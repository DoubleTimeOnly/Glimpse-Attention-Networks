from typing import Optional
import torch.nn as nn
import torch


class CoreNetwork(nn.Module):
    """
    The core RNN network that maintains internal state.
    """
    def __init__(self, glimpse_hidden: int = 256, hidden_size: int = 256):
        super(CoreNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(glimpse_hidden, hidden_size)
        
    def forward(self, glimpse_repr: torch.Tensor, 
                hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update RNN state with glimpse representation.
        
        Args:
            glimpse_repr: glimpse representation (B, glimpse_hidden)
            hidden_state: previous (h, c) state
            
        Returns:
            h: new hidden state (B, hidden_size)
            (h, c): new hidden and cell states
        """
        if hidden_state is None:
            batch_size = glimpse_repr.size(0)
            h = torch.zeros(batch_size, self.hidden_size, device=glimpse_repr.device)
            c = torch.zeros(batch_size, self.hidden_size, device=glimpse_repr.device)
            hidden_state = (h, c)
            
        h, c = self.rnn(glimpse_repr, hidden_state)
        return h, (h, c)
