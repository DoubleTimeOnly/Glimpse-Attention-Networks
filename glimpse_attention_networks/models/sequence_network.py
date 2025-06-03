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
        self.rnn1 = nn.LSTMCell(glimpse_hidden, hidden_size)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size)
        
    def forward(
        self,
        r1_input: torch.Tensor,
        states_1: tuple[torch.Tensor, torch.Tensor],
        states_2: tuple[torch.Tensor, torch.Tensor],
        first_step: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update RNN state with glimpse representation.
        
        Args:
            glimpse_repr: glimpse representation (B, glimpse_hidden)
            hidden_state: previous (h, c) state
            states_1: previous (h, c) state for the first LSTM layer
            states_2: previous (h, c) state for the second LSTM layer
            
        Returns:
            h: new hidden state (B, hidden_size)
            (h, c): new hidden and cell states
        """
        h1, c1 =  self.rnn1(r1_input, states_1)

        if first_step:
            r2_input = torch.zeros(
                (r1_input.shape[0], self.hidden_size), dtype=torch.float32, device=r1_input.device
            )
        else:
            r2_input = h1

        h2, c2 = self.rnn2(r2_input, states_2)

        return (h1, c1), (h2, c2)

