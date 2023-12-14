import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MixerBlock(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x

if __name__ == '__main__':
    net = MixerBlock(2, 3, 4, 4)
    dummy = torch.randn(4,2,3)
    #[B,L,C]
    layernorm = nn.LayerNorm(3)
    print(net(dummy))
    print(layernorm(net(dummy)))