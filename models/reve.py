import numpy as np
import torch
from einops import rearrange
from models.flash_vit_utils import Transformer
from torch import nn

from models.patch_embed import FourierEmb4D, expand_with_time_dimension, mlp_pos_embedding, patch_embedding


class ReveEncoder(nn.Module):
    def __init__( 
        self,
        patch_size=200,
        overlap_size=40,
        noise_ratio=0.005,
        embed_dim=256,
        depth=6,
        heads=6,
        mlp_dim_ratio=2,
        dim_head=64,
        use_flash=True,
        geglu=True,
    ):
        super().__init__()

        self.mlp_dim_ratio = mlp_dim_ratio
        mlp_dim = int(self.mlp_dim_ratio * embed_dim)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.noise_ratio = noise_ratio
        self.to_patch_embedding = patch_embedding(embed_dim, patch_size)
        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, use_flash, geglu)
        self.dropout_classif = nn.Dropout(0.0)
        self.increment_time = 0.1  # used for scaling to the time unit
        self.add_time_patch = expand_with_time_dimension
        self.fourier4d = FourierEmb4D(embed_dim, increment_time=self.increment_time)
        self.mlp4d = mlp_pos_embedding(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, eeg, pos=None, return_output=False):
        device = eeg.device
        eeg = eeg.float()
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size)
        b, c, h, p = patches.shape
        if self.training:  # add noise to the positions
            pos += torch.from_numpy(np.random.normal(loc=0, scale=self.noise_ratio, size=(c, 3))).to(device)
        pos = self.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))
        x = rearrange(self.to_patch_embedding(patches), "b c h e -> b (c h) e", c=c, h=h, e=self.embed_dim) + pos_embed
        x = self.transformer(x, return_output)
        return x
