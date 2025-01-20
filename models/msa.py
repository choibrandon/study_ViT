# 1/20 여기부터 해야함

import torch
import torch.nn as nn

class MSA(nn.Module):
    def __init__(self, dim =192, num_heads =12, qkv_bias=False,attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
