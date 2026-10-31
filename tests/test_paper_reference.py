"""P0 falsification, numeric oracle, parity and causal decoding tests."""
import pytest
import torch
from grassmann_edge.paper_reference import (
    PaperGrassmannBlock, PaperGrassmannLM,
    plucker, plucker_scalar_oracle, receptive_field)


def test_plucker_independent_double_precision_oracle():
    torch.manual_seed(11)
    u = torch.randn(2, 5, 9, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 5, 9, dtype=torch.float64, requires_grad=True)
    actual, oracle = plucker(u, v), plucker_scalar_oracle(u, v)
    torch.testing.assert_close(actual, oracle, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), (u, v)),
        torch.autograd.grad(oracle.square().sum(), (u, v)),
        atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("offsets", [(1,), (1, 2, 4), (2, 6)])
def test_causal_prefix_invariance(offsets):
    torch.manual_seed(2026)
    layer = PaperGrassmannBlock(dim=16, rank=5, offsets=offsets).double().eval()
    x = torch.randn(2, 14, 16, dtype=torch.float64)
    changed = x.clone()
    changed[:, 7:] += torch.randn_like(changed[:, 7:])
    with torch.no_grad():
        original, mutated = layer(x), layer(changed)
    torch.testing.assert_close(original[:, :7], mutated[:, :7], atol=0, rtol=0)


def test_literal_future_indexing_leaks():
    torch.manual_seed(2026)
    layer = PaperGrassmannBlock(dim=16, rank=5, offsets=(1,),
                                direction="paper_literal").double().eval()
    x = torch.randn(1, 12, 16, dtype=torch.float64)
    changed = x.clone()
    changed[:, 6] += 2
    diff = (layer(x)[:, 5] - layer(changed)[:, 5]).abs().max().item()
    assert diff > 1e-8, f"Unexpected absence of future influence: {diff}"


def test_literal_future_gradient_nonzero_and_causal_gradient_zero():
    torch.manual_seed(17)
    x = torch.randn(1, 9, 16, dtype=torch.float64, requires_grad=True)
    future = PaperGrassmannBlock(16, 5, (1,), "paper_literal").double()
    past = PaperGrassmannBlock(16, 5, (1,), "past_causal").double()
    for model in (future, past):
        val = model(x)[:, 4].square().sum()
        grad, = torch.autograd.grad(val, x)
        magnitude = grad[:, 5:].abs().max().item()
        if model.direction == "paper_literal":
            assert magnitude > 1e-10
        else:
            assert magnitude == 0


@pytest.mark.parametrize("direction", ["past_causal", "paper_literal"])
def test_boundary_policy_and_finite_gradients(direction):
    block = PaperGrassmannBlock(8, 3, (1, 2, 8),
                                 direction).double().eval()
    x = torch.randn(2, 1, 8, dtype=torch.float64, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())


@pytest.mark.parametrize("length", [1, 2, 5, 9, 17])
def test_full_vs_incremental_causal_lm(length):
    torch.manual_seed(19)
    model = PaperGrassmannLM(vocab=33, dim=16, layers=2,
                             rank=5, context=32, offsets=(1, 2, 4)).eval()
    ids = torch.randint(0, 33, (2, length))
    with torch.inference_mode():
        full = model(ids)
        incremental = model.forward_incremental(ids)
    torch.testing.assert_close(full, incremental, atol=2e-6, rtol=2e-6)


def test_future_variant_refuses_incremental_decode():
    lm = PaperGrassmannLM(direction="paper_literal")
    with pytest.raises(ValueError, match="cannot decode causally"):
        lm.forward_incremental(torch.ones(1, 2, dtype=torch.long))


def test_exact_receptive_field_bound():
    assert receptive_field(6, (1, 2, 4, 8, 12, 16)) == 97
    assert receptive_field(2, (1, 4)) == 9


def test_autoregressive_model_prefix_invariance():
    torch.manual_seed(20)
    model = PaperGrassmannLM(vocab=33, dim=16, layers=3, rank=5,
                             context=32, offsets=(1, 2, 4)).eval()
    ids = torch.randint(0, 33, (2, 32))
    future = ids.clone()
    future[:, 16:] = (future[:, 16:] + 11) % 33
    with torch.inference_mode():
        a, b = model(ids), model(future)
    torch.testing.assert_close(a[:, :16], b[:, :16], rtol=0, atol=0)
