import torch
from torch.nn import functional as F
from grassmann_edge import TinyLM, delayed_copy_batch


def test_train_step_for_all_models():
    torch.manual_seed(13)
    batch = delayed_copy_batch(3, 16, 16, lag=4, seed=120)
    for kind in ("grassmann", "transformer", "local"):
        model = TinyLM(kind=kind, vocab=16, dim=16, layers=1,
                       rank=4, offsets=(1, 2, 4), context=16)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001)
        old = model.embed.weight.detach().clone()
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 16), batch[:, 1:].reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        assert torch.isfinite(loss)
        assert not torch.equal(old, model.embed.weight)


def test_synthetic_seed_reproducibility():
    a = delayed_copy_batch(10, 64, 23, lag=11, seed=98)
    b = delayed_copy_batch(10, 64, 23, lag=11, seed=98)
    assert torch.equal(a, b)
