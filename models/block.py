import torch.nn as nn
import torch
from .msa import MSA
from .mlp import MLP

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop =0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
if __name__ == '__main__':
    def main():
        block = Block(dim=192, num_heads=12)
        sample_input = torch.randn(1, 10, 192)
        output = block(sample_input)
        print("Block output shape:", output.shape)
    main()