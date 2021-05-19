import torch
from torch import nn, einsum
import einops


class MlpBlock(nn.Module):
    def __init__(self, dim, inter_dim, dropout_ratio):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)


class MixerLayer(nn.Module):
    def __init__(self, 
                 inter_dim, 
                 token_dim, 
                 token_inter_dim, 
                 channel_inter_dim, 
                 dropout_ratio):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(inter_dim)
        self.Mlp_token = MlpBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(inter_dim)
        self.Mlp_channel = MlpBlock(inter_dim, channel_inter_dim, dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.Mlp_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.Mlp_channel(z)
        out = x + y + z
        return out


class Mlp_Mixer(nn.Module):
    def __init__(self,
                 dim,
                 inter_dim,
                 token_inter_dim,
                 channel_inter_dim,
                 img_size: list,
                 patch_size: list,
                 num_block,
                 num_class,
                 dropout_ratio=0):
        super().__init__()
        self.token_dim = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.embedding = nn.Conv2d(dim, inter_dim, stride=patch_size, kernel_size=patch_size, padding=0)
        self.mixer_blocks = nn.ModuleList(
            [MixerLayer(inter_dim, self.token_dim, token_inter_dim, channel_inter_dim, dropout_ratio) for _ in
             range(num_block)])
        self.head_layer_norm = nn.LayerNorm(inter_dim)
        self.fc = nn.Linear(self.token_dim, num_class)

    def forward(self, x):
        x = self.embedding(x)
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        for num_layers in self.mixer_blocks:
            x = num_layers(x)

        x = self.head_layer_norm(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    t1 = torch.rand(1, 3, 128, 128)
    net = Mlp_Mixer(3, 32, 32, 32, (128, 128), (16, 16), 2, 6, 0)
    out = net(t1)
    # print(out.shape)
