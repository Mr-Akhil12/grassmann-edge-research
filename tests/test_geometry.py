import pytest
import torch
from grassmann_edge import GrassmannMix, wedge


def test_wedge_antisymmetry_and_degeneracy():
    torch.manual_seed(7)
    u, v = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    i, j = torch.triu_indices(16, 16, offset=1)
    assert torch.allclose(wedge(u, v, i, j), -wedge(v, u, i, j), atol=1e-6)
    assert torch.allclose(wedge(u, u * 3.0, i, j), torch.zeros_like(wedge(u, u, i, j)), atol=2e-6)


def test_gradient_finite():
    torch.manual_seed(8)
    block = GrassmannMix(dim=16, rank=4, offsets=(1, 2)).double()
    x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    block(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


def test_bad_configuration_rejected():
    with pytest.raises(ValueError):
        GrassmannMix(rank=1)
    with pytest.raises(ValueError):
        GrassmannMix(offsets=(1, -2))
