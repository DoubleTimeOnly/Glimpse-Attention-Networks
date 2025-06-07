from typing import Optional, Tuple
import torch.nn as nn
import torch
import math


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


class TransformerEncoderDecoder(nn.Module):
    """
    Transformer-based encoder-decoder network that processes sequences of glimpses.
    The encoder processes input tokens and the decoder generates output tokens
    using cross-attention with the encoder's output.
    """
    def __init__(
        self,
        glimpse_hidden: int = 256,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, dropout, max_seq_len)
        
        # Project glimpse representation to transformer dimension
        self.glimpse_proj = nn.Linear(glimpse_hidden, hidden_size)

        # project location to transformer dimension
        self.location_proj = nn.Linear(2, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Project transformer output to hidden size
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Learnable query embeddings for the decoder
        self.query_embed = nn.Parameter(torch.randn(1, hidden_size))

        self.class_embed = nn.Parameter(torch.randn(1, hidden_size))

        self.glimpse_start_token = nn.Parameter(torch.randn(1, hidden_size))
        self.location_start_token = nn.Parameter(torch.randn(1, 2))
        
    def forward(
        self,
        glimpse_context: list[torch.Tensor],
        global_context: torch.Tensor,
        previous_queries: list[torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process glimpse through transformer encoder-decoder network.
        
        Args:
            context: list of glimpse representations for batch.
            There is one element for each glimpse taken.
            Each element is (N, glimpse_hidden) and represents the glimpse vector for that step
            previous_queries: list of previous query vectors for batch.
            Each element is (N, hidden_size) and represents the query vector for that step
            first_step: whether this is the first step
            
        Returns:
            tuple of state tuples to maintain compatibility with RNN interface
        """
        batch_size = global_context.size(0)
        if len(glimpse_context) > 0:

            # N, num_glimpses_taken+1, glimpse_hidden
            stacked_context = torch.stack(glimpse_context, dim=1)
            
            # Project glimpse to transformer dimension
            x = self.glimpse_proj(stacked_context)  # (N, num_glimpses_taken+1, hidden_size)
            
            # Process through encoder
            mask = self.get_causal_attention_mask(x.size(1), device=x.device)
            memory = self.encoder(
                x,
                mask=mask,
            )  # (N, num_glimpses_taken+1, hidden_size)
        else:
            # make memory just class token
            # N, 1, hidden_size
            memory = self.class_embed.expand(batch_size, 1, -1)

        class_token = memory[:, -1, :]

        # avoid adding global_context to encoder to keep glimpses separate
        global_context = self.glimpse_proj(global_context)
        memory = torch.cat([global_context.unsqueeze(1), memory], dim=1)
        
        # Create query embeddings for decoder
        # N, num_glimpses_taken, hidden_size
        # query = self.query_embed.expand(batch_size, -1)  # (N, hidden_size)
        all_queries = torch.stack(previous_queries, dim=1)
        all_queries = self.location_proj(all_queries)
        all_queries = self.pos_encoding(all_queries)
        
        # Process through decoder
        # each location query can only attend to previous locations
        tgt_mask = self.get_causal_attention_mask(all_queries.size(1), device=all_queries.device)
        # let all queries attend to global context
        mem_mask = torch.cat([
            torch.zeros(tgt_mask.shape[0], 1, device=all_queries.device), tgt_mask
        ], dim=1)
        output = self.decoder(
            tgt=all_queries,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=mem_mask,
            tgt_is_causal=True,
            memory_is_causal=True,
        )  # (N, num_glimpses_taken+1, hidden_size)

        # output = self.output_proj(output)  # (N, num_glimpses_taken+1, hidden_size)
        # Get only the most recent query output
        next_loc_vector = output[:, -1, :]  # (N, hidden_size)
        
        # where and what vectors
        return next_loc_vector, class_token
    
    def get_causal_attention_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = torch.tril(torch.ones(sz, sz, device=device))
        mask[mask==0] = float('-inf')
        mask[mask==1] = float(0.0)
        return mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
