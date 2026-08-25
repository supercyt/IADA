"""Masked window/grid attention blocks used by CoBEVT.

Adapted from HEAL's CoBEVT implementation for the homogeneous LiDAR-only
fusion interface used in this repository.
"""

import torch
from einops import rearrange
from torch import einsum, nn

from opencood.models.sub_modules.base_transformer import FeedForward


class PreNormResidual(nn.Module):
    def __init__(self, dim, function):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.function = function

    def forward(self, features, **kwargs):
        return self.function(self.norm(features), **kwargs) + features


class Attention(nn.Module):
    """Local 3-D attention over agents and spatial window tokens."""

    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.0,
        agent_size=6,
        window_size=7,
    ):
        super().__init__()
        if dim % dim_head != 0:
            raise ValueError("CoBEVT input_dim must be divisible by dim_head")
        self.heads = dim // dim_head
        self.scale = dim_head**-0.5
        self.window_size = (agent_size, window_size, window_size)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        relative_count = 1
        for size in self.window_size:
            relative_count *= 2 * size - 1
        self.relative_position_bias_table = nn.Embedding(
            relative_count, self.heads
        )

        coordinates = torch.stack(
            torch.meshgrid(
                *(torch.arange(size) for size in self.window_size),
                indexing="ij",
            )
        )
        flattened = torch.flatten(coordinates, 1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        for axis, size in enumerate(self.window_size):
            relative[:, :, axis] += size - 1
        relative[:, :, 0] *= (
            (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1)
        )
        relative[:, :, 1] *= 2 * self.window_size[2] - 1
        self.register_buffer(
            "relative_position_index", relative.sum(-1), persistent=True
        )

    def forward(self, features, mask=None):
        batch, agents, height, width, win_h, win_w, _ = features.shape
        features = rearrange(
            features, "b l x y w1 w2 d -> (b x y) (l w1 w2) d"
        )
        query, key, value = self.to_qkv(features).chunk(3, dim=-1)
        query, key, value = map(
            lambda tensor: rearrange(
                tensor, "b n (h d) -> b h n d", h=self.heads
            ),
            (query, key, value),
        )
        similarity = einsum(
            "b h i d, b h j d -> b h i j", query * self.scale, key
        )
        bias = self.relative_position_bias_table(
            self.relative_position_index
        )
        similarity = similarity + rearrange(bias, "i j h -> h i j")

        if mask is not None:
            mask = rearrange(
                mask,
                "b x y w1 w2 e l -> (b x y) e (l w1 w2)",
            ).unsqueeze(1)
            similarity = similarity.masked_fill(mask == 0, -torch.inf)

        attention = self.attend(similarity)
        output = einsum("b h i j, b h j d -> b h i d", attention, value)
        output = rearrange(
            output,
            "b h (l w1 w2) d -> b l w1 w2 (h d)",
            l=agents,
            w1=win_h,
            w2=win_w,
        )
        output = self.to_out(output)
        return rearrange(
            output,
            "(b x y) l w1 w2 d -> b l x y w1 w2 d",
            b=batch,
            x=height,
            y=width,
        )


class SwapFusionBlockMask(nn.Module):
    """CoBEVT window attention followed by sparse grid attention."""

    def __init__(
        self,
        input_dim,
        mlp_dim,
        dim_head,
        window_size,
        agent_size,
        drop_out,
    ):
        super().__init__()
        self.window_size = int(window_size)
        attention_args = (
            input_dim,
            dim_head,
            drop_out,
            agent_size,
            window_size,
        )
        self.window_attention = PreNormResidual(
            input_dim, Attention(*attention_args)
        )
        self.window_ffd = PreNormResidual(
            input_dim, FeedForward(input_dim, mlp_dim, drop_out)
        )
        self.grid_attention = PreNormResidual(
            input_dim, Attention(*attention_args)
        )
        self.grid_ffd = PreNormResidual(
            input_dim, FeedForward(input_dim, mlp_dim, drop_out)
        )

    def forward(self, features, mask):
        window = self.window_size
        window_mask = rearrange(
            mask,
            "b (x w1) (y w2) e l -> b x y w1 w2 e l",
            w1=window,
            w2=window,
        )
        features = rearrange(
            features,
            "b m d (x w1) (y w2) -> b m x y w1 w2 d",
            w1=window,
            w2=window,
        )
        features = self.window_attention(features, mask=window_mask)
        features = self.window_ffd(features)
        features = rearrange(
            features, "b m x y w1 w2 d -> b m d (x w1) (y w2)"
        )

        grid_mask = rearrange(
            mask,
            "b (w1 x) (w2 y) e l -> b x y w1 w2 e l",
            w1=window,
            w2=window,
        )
        features = rearrange(
            features,
            "b m d (w1 x) (w2 y) -> b m x y w1 w2 d",
            w1=window,
            w2=window,
        )
        features = self.grid_attention(features, mask=grid_mask)
        features = self.grid_ffd(features)
        return rearrange(
            features, "b m x y w1 w2 d -> b m d (w1 x) (w2 y)"
        )


__all__ = ["Attention", "SwapFusionBlockMask"]
