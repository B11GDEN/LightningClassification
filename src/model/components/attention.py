import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
            self, dim: int, num_heads: int = 8,
            qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attn)

        x = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


kernels = {
    'l1': lambda x: F.normalize(x, p=1.0, dim=-1),
    'l2': lambda x: F.normalize(x, p=2.0, dim=-1),
    'tanh': lambda x: F.tanh(x),
    'softmax': lambda x: F.softmax(x, dim=-1),
    'sigmoid': lambda x: F.sigmoid(x),
}


class LinearAttention(nn.Module):
    def __init__(
            self, dim: int, num_heads: int = 8,
            qkv_bias: bool = True, kv_drop: float = 0., proj_drop: float = 0.,
            q_kernel: str = 'l2', k_kernel: str = 'l2',
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_drop = nn.Dropout(kv_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_kernel = kernels[q_kernel]
        self.k_kernel = kernels[k_kernel]

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q, k = self.q_kernel(q), self.k_kernel(k)

        kv = (k.transpose(-2, -1) @ v)
        kv = self.kv_drop(kv)

        x = (q @ kv).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

if __name__ == "__main__":
    attn = Attention(
        dim=768, num_heads=12,
    )
    inp = torch.FloatTensor