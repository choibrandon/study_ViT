import torch
import torch.nn as nn
from .embedding import EmbeddingLayer
from .block import Block

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=12,
                 num_heads=12, mlp_ratio=2., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.path_embed = EmbeddingLayer(in_chans, embed_dim, img_size, patch_size)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.path_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)[:, 0]
        return x
    
if __name__ == '__main__':
    def main():
        model = ViT()
        sample_input = torch.randn(1, 3, 32, 32)
        output = model(sample_input)
        print("ViT output shape:", output.shape)
    main()