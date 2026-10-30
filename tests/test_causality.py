import pytest
import torch
from grassmann_edge import TinyLM, GrassmannMix


@pytest.mark.parametrize("kind", ["grassmann", "transformer", "local"])
def test_full_lm_prefix_unchanged_by_future_suffix(kind):
    torch.manual_seed(2026)
    model = TinyLM(kind=kind, vocab=23, dim=32, layers=3, rank=8,
                   offsets=(1, 2, 4, 8), context=32).eval()
    x = torch.randint(0, 23, (2, 32))
    changed = x.clone()
    changed[:, 18:] = (changed[:, 18:] + 1) % 23
    with torch.inference_mode():
        lhs, rhs = model(x), model(changed)
    assert torch.allclose(lhs[:, :18], rhs[:, :18], atol=1e-6, rtol=1e-6)


def test_future_pairing_is_not_causal():
    torch.manual_seed(2026)
    block = GrassmannMix(dim=16, rank=4, offsets=(1,), direction="future_diagnostic").eval()
    x = torch.randn(1, 16, 16)
    changed = x.clone()
    changed[:, 8:] += 5
    with torch.inference_mode():
        assert (block(x)[:, :8] - block(changed)[:, :8]).abs().max() > 1e-4


def test_causal_mix_handles_sequence_of_length_one():
    block = GrassmannMix(dim=16, rank=4, offsets=(1, 2)).eval()
    assert torch.isfinite(block(torch.randn(2, 1, 16))).all()
