"""Seedable synthetic delayed-copy sequence for model correctness diagnostics."""
import torch


def delayed_copy_batch(batch, length, vocab, lag, seed, copy_probability=0.9):
    if not 1 <= lag <= length or not 0 <= copy_probability <= 1 or vocab < 2:
        raise ValueError("Bad parameters")
    g = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(vocab, (batch, length + 1), generator=g)
    for t in range(lag, length + 1):
        keep = torch.rand((batch,), generator=g) < copy_probability
        tokens[:, t] = torch.where(keep, tokens[:, t - lag], tokens[:, t])
    return tokens
