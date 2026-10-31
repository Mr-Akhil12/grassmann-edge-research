import pytest
import torch
from grassmann_edge import TinyLM
from grassmann_edge.inference_cache import incremental_tiny, model_state_bytes


@pytest.mark.parametrize("kind", ["transformer", "local"])
@pytest.mark.parametrize("length", [1, 4, 9, 18])
def test_incremental_equals_full_reference(kind, length):
    torch.manual_seed(999)
    model = TinyLM(kind=kind, vocab=29, dim=32, layers=2,
                   rank=8, offsets=(1, 2, 4), context=32).eval()
    ids = torch.randint(0, 29, (2, length))
    with torch.inference_mode():
        full = model(ids)
        cached = incremental_tiny(model, ids)
    torch.testing.assert_close(full, cached, atol=3e-5, rtol=3e-5)


def test_cache_memory_growth_structure():
    transformer = TinyLM(kind="transformer", dim=32, layers=2, context=128)
    local = TinyLM(kind="local", dim=32, layers=2, rank=8,
                   offsets=(1, 2, 4, 8), context=128)
    assert model_state_bytes(transformer, 128) > model_state_bytes(transformer, 16)
    assert model_state_bytes(local, 128) == model_state_bytes(local, 16)
