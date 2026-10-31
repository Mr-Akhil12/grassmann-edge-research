"""Paper-literal and corrected causal Grassmann layers.

arXiv:2512.19428v1, Section 3.2, equations for reduction, pairing,
Pluecker coordinates, projection, gated fusion, norm/dropout and FFN.

The manuscript explicitly pairs z_t with z_(t + delta) and calls it
"causal". We preserve that behavior only in paper_literal mode; this
leaks future tokens in conventional autoregressive prediction.

No claim that this module reproduces the paper's reported LM metrics.
Boundary policy when there are no eligible offsets: g_t = zeros.
Normalization is enabled using max(norm, eps) as in Section 3.2.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def plucker(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pairwise i<j exterior products: u_i v_j - u_j v_i."""
    if u.shape != v.shape or u.shape[-1] < 2:
        raise ValueError("u and v must have equal shapes with rank >= 2")
    i, j = torch.triu_indices(u.shape[-1], u.shape[-1], 1, device=u.device)
    return u[..., i] * v[..., j] - u[..., j] * v[..., i]


def plucker_scalar_oracle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Independent scalar-loop oracle, intentionally not optimized."""
    if u.shape != v.shape or u.shape[-1] < 2:
        raise ValueError("Invalid pair")
    rank = u.shape[-1]
    return torch.stack(
        [u[..., i] * v[..., j] - u[..., j] * v[..., i]
         for i in range(rank) for j in range(i + 1, rank)], dim=-1)


class PaperGrassmannBlock(nn.Module):
    """Section 3.2 normalized Grassmann block with explicit direction.

    paper_literal follows t+delta as written (non-causal).
    past_causal replaces future with t-delta, retaining equation order.
    """
    def __init__(self, dim=64, rank=8, offsets=(1, 2, 4, 8),
                 direction="past_causal", dropout=0.0, eps=1e-8):
        super().__init__()
        if direction not in ("past_causal", "paper_literal"):
            raise ValueError("Unknown direction")
        if dim < 1 or rank < 2 or not offsets or min(offsets) < 1 or len(set(offsets)) != len(offsets):
            raise ValueError("Invalid dimensions or offsets")
        if not 0 <= dropout < 1 or eps <= 0:
            raise ValueError("Invalid dropout or epsilon")
        self.dim, self.rank, self.offsets = dim, rank, tuple(offsets)
        self.direction, self.eps = direction, eps
        self.reduce = nn.Linear(dim, rank)
        self.project = nn.Linear(rank * (rank - 1) // 2, dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.mix_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim))
        self.ff_norm = nn.LayerNorm(dim)

    def geometric_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        length = x.shape[1]
        total = x.new_zeros(x.shape)
        counts = x.new_zeros((1, length, 1))
        for delta in self.offsets:
            if delta >= length:
                continue
            if self.direction == "past_causal":
                here, other, idx = z[:, delta:], z[:, :-delta], slice(delta, None)
            else:
                here, other, idx = z[:, :-delta], z[:, delta:], slice(None, -delta)
            p = plucker(here, other)
            p = p / torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp_min(self.eps)
            total[:, idx] += self.project(p)
            counts[:, idx] += 1
        return total / counts.clamp_min(1)

    def _finish(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((x, g), dim=-1)))
        mix = gate * x + (1 - gate) * g
        normed = self.dropout(self.mix_norm(mix))
        return self.ff_norm(normed + self.ff(normed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError("Expected (batch, length, dim)")
        return self._finish(x, self.geometric_features(x))

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, cache: list[torch.Tensor]):
        """Incremental past-only step with bounded rank-vector state.

        x_t: (batch, 1, dim); cache is previous low-rank z's in order.
        Returns (output, updated_cache). Never usable in paper_literal mode.
        """
        if self.direction != "past_causal":
            raise ValueError("Paper-literal future pairing cannot decode causally")
        if x_t.ndim != 3 or x_t.size(1) != 1 or x_t.size(-1) != self.dim:
            raise ValueError("Expected (batch, 1, dim)")
        z_t = self.reduce(x_t)
        g = x_t.new_zeros(x_t.shape)
        num = 0
        for delta in self.offsets:
            if len(cache) >= delta:
                p = plucker(z_t, cache[-delta])
                p = p / torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp_min(self.eps)
                g = g + self.project(p)
                num += 1
        if num:
            g = g / num
        keep = max(self.offsets)
        updated = (cache + [z_t])[-keep:]
        return self._finish(x_t, g), updated


class PaperGrassmannLM(nn.Module):
    def __init__(self, vocab=64, dim=64, layers=2, rank=8,
                 context=128, offsets=(1, 2, 4, 8),
                 direction="past_causal"):
        super().__init__()
        self.context, self.direction = context, direction
        self.token = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList([
            PaperGrassmannBlock(dim, rank, offsets, direction)
            for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids: torch.Tensor):
        if ids.ndim != 2 or ids.size(1) > self.context:
            raise ValueError("Invalid input ids")
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.token(ids) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))

    @torch.no_grad()
    def forward_incremental(self, ids: torch.Tensor):
        """Reference incremental decoding, each layer caches reduced past z."""
        if self.direction != "past_causal":
            raise ValueError("Future-looking variant cannot decode causally")
        if ids.ndim != 2 or ids.size(1) > self.context:
            raise ValueError("Invalid input ids")
        caches = [[] for _ in self.blocks]
        output = []
        for t in range(ids.size(1)):
            x = self.token(ids[:, t:t+1]) + self.position(
                torch.tensor([t], device=ids.device))
            for i, block in enumerate(self.blocks):
                x, caches[i] = block.step(x, caches[i])
            output.append(self.head(self.final_norm(x)))
        return torch.cat(output, dim=1)


def receptive_field(layers: int, offsets: tuple[int, ...]) -> int:
    """Maximum past-inclusive theoretical span under fixed offsets per layer."""
    if layers < 1 or not offsets or min(offsets) < 1:
        raise ValueError("Invalid receptive field configuration")
    return 1 + layers * max(offsets)
