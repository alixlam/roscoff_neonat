import math

import torch
from torch import nn


def expand_with_time_dimension(pos, num_patches):
    """
    Expand the position tensor by adding a time dimension, handling batched data.

    Args:
    - pos (Tensor): Input tensor of shape (B, C, 3), where B is the batch size,
    C is the number of channels, and 3 represents x, y, z.
    - num_patches (int): The number of time patches.
    - time_increment (float): The increment for the time values (default is 1.0).

    Returns:
    - Tensor: Output tensor of shape (B, C * num_patches, 4), where each position is repeated with each time value.
    """
    B, C, _ = pos.shape
    # Repeat each position for each time step
    pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)  # Shape: (B, C, num_patches, 3)
    # Generate time values with the specified increment
    time_values = torch.arange(0, num_patches, 1, device=pos.device).float()  # Shape: (num_patches,)
    time_values = time_values.view(1, 1, num_patches, 1).expand(B, C, num_patches, 1)  # Shape: (B, C, num_patches, 1)
    # Concatenate the repeated positions with the time values along the last dimension
    pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)  # Shape: (B, C, num_patches, 4)
    # Reshape to (B, C * num_patches, 4)
    pos_with_time = pos_with_time.view(B, C * num_patches, 4)

    return pos_with_time


class FourierEmb4D(nn.Module):
    """
    Fourier positional embedding for 4D positions (x, y, z, t).
    This version allows for a reduced number of frequencies (n_freqs),
    and ensures the output embedding has the specified dimension.
    """

    def __init__(self, dimension: int = 512, increment_time=0.1, margin: float = 0.4):
        super().__init__()
        self.increment_time = increment_time
        assert dimension%2 == 0, 'Dimension must be even'
        self.dimension = dimension
        self.margin = margin
        self.nfreqs = math.ceil((dimension / 2) ** (1/4))

    def forward(self, positions_):
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *O, D = positions.shape
        n_freqs = self.nfreqs
        freqs_w = torch.arange(n_freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x + positions[..., 1] * p_y + positions[..., 2] * p_z + positions[..., 3] * p_w
        ).view(*O, -1)
        if self.dimension != 2*(n_freqs**4):
            _, _, hd = loc.shape
            diff = hd - self.dimension // 2
            loc = loc[:, :, :-diff]
        emb = torch.cat(
            [
                torch.cos(loc),
                torch.sin(loc),
            ],
            dim=-1,
        )
        return emb


def patch_embedding(embed_dim, patch_size):
    to_patch_embedding = nn.Sequential(nn.Linear(patch_size, embed_dim))
    return to_patch_embedding


def mlp_pos_embedding(embed_dim):
    mlp_pos_embedding = nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))
    return mlp_pos_embedding
