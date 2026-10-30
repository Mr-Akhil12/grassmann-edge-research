"""Causally corrected geometric mixing research pilot.

This is not a paper-exact implementation of arXiv:2512.19428.
In particular we use t-offset, not the paper's t+offset pairing.
Three model variants are intentionally not parameter-matched yet.
"""
import torch
from torch import nn
from torch.nn import functional as F


def wedge(u, v, i, j):
    """Pluecker coordinates, u_i*v_j - u_j*v_i."""
    return u[..., i] * v[..., j] - u[..., j] * v[..., i]


class GrassmannMix(nn.Module):
    def __init__(self, dim=64, rank=16, offsets=(1, 2, 4, 8),
                 dropout=0.0, direction="past"):
        super().__init__()
        if rank < 2 or not offsets or any(k < 1 for k in offsets):
            raise ValueError("Rank >=2 and positive offsets are required")
        if len(set(offsets)) != len(offsets):
            raise ValueError("Offsets must be unique")
        if direction not in ("past", "future_diagnostic"):
            raise ValueError("Bad direction")
        self.offsets = tuple(offsets)
        self.direction = direction
        self.reduce = nn.Linear(dim, rank)
        self.project = nn.Linear(rank * (rank - 1) // 2, dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.mix_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim))
        self.ff_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        i, j = torch.triu_indices(rank, rank, offset=1)
        self.register_buffer("i", i, persistent=False)
        self.register_buffer("j", j, persistent=False)

    def geometry(self, x):
        z = self.reduce(x)
        length = x.shape[1]
        accum = torch.zeros_like(x)
        counts = x.new_zeros((1, length, 1))
        for offset in self.offsets:
            if offset >= length:
                continue
            if self.direction == "past":
                center, neighbor, loc = z[:, offset:], z[:, :-offset], slice(offset, None)
            else:
                center, neighbor, loc = z[:, :-offset], z[:, offset:], slice(None, -offset)
            features = wedge(center, neighbor, self.i, self.j)
            features = F.normalize(features, p=2, dim=-1, eps=1e-8)
            accum[:, loc] += self.project(features)
            counts[:, loc] += 1.0
        return accum / counts.clamp_min(1.0)

    def forward(self, x):
        geo = self.geometry(x)
        alpha = self.gate(torch.cat((x, geo), dim=-1)).sigmoid()
        h = self.mix_norm(self.dropout(alpha * x + (1 - alpha) * geo))
        return self.ff_norm(h + self.dropout(self.ff(h)))


class LocalMix(nn.Module):
    """Past-looking non-geometric control; parameters not yet matched."""
    def __init__(self, dim=64, rank=16, offsets=(1, 2, 4, 8), dropout=0.0):
        super().__init__()
        self.offsets = tuple(offsets)
        self.reduce = nn.Linear(dim, rank)
        self.project = nn.Linear(2 * rank, dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.mix_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim))
        self.ff_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.reduce(x)
        length = x.shape[1]
        accum = torch.zeros_like(x)
        counts = x.new_zeros((1, length, 1))
        for offset in self.offsets:
            if offset >= length:
                continue
            features = torch.cat((z[:, offset:], z[:, :-offset]), dim=-1)
            accum[:, offset:] += self.project(features)
            counts[:, offset:] += 1.0
        geo = accum / counts.clamp_min(1.0)
        alpha = self.gate(torch.cat((x, geo), dim=-1)).sigmoid()
        h = self.mix_norm(self.dropout(alpha * x + (1 - alpha) * geo))
        return self.ff_norm(h + self.dropout(self.ff(h)))


class TransformerMix(nn.Module):
    """Small causal SDPA reference (no KV-cache in pilot)."""
    def __init__(self, dim=64, heads=4, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.mix_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim))
        self.ff_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.heads, d // self.heads)
        q, k, v = (qkv[:, :, j].transpose(1, 2) for j in range(3))
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0)
        h = self.mix_norm(x + self.dropout(
            self.out(attention.transpose(1, 2).reshape(b, l, d))))
        return self.ff_norm(h + self.dropout(self.ff(h)))


class TinyLM(nn.Module):
    def __init__(self, kind="grassmann", vocab=32, dim=64, layers=2,
                 rank=8, offsets=(1, 2, 4, 8), context=128, heads=4):
        super().__init__()
        if kind not in ("grassmann", "local", "transformer"):
            raise ValueError("Unknown model kind")
        self.kind = kind
        self.context = context
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList([
            GrassmannMix(dim, rank, offsets) if kind == "grassmann" else
            LocalMix(dim, rank, offsets) if kind == "local" else
            TransformerMix(dim, heads)
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids):
        b, length = ids.shape
        if length > self.context:
            raise ValueError("Exceeded configured context")
        x = self.embed(ids) + self.pos(torch.arange(length, device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
