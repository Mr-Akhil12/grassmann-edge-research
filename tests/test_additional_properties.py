import torch
from grassmann_edge import wedge, TinyLM


def test_pluecker_lagrange_identity():
    torch.manual_seed(39)
    u = torch.randn(3, 12, dtype=torch.float64)
    v = torch.randn(3, 12, dtype=torch.float64)
    i, j = torch.triu_indices(12, 12, 1)
    p = wedge(u, v, i, j)
    lhs = p.square().sum(dim=-1)
    rhs = u.square().sum(-1) * v.square().sum(-1) - (u * v).sum(-1).square()
    torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_no_future_input_gradient_into_prefix_output():
    torch.manual_seed(40)
    for kind in ("grassmann", "transformer", "local"):
        model = TinyLM(kind, vocab=19, dim=16, layers=2,
                       rank=4, offsets=(1, 2), context=16)
        x = torch.randn(1, 16, 16, requires_grad=True)
        value = model.blocks[0](x)[:, 5].square().sum()
        grad, = torch.autograd.grad(value, x)
        assert torch.equal(grad[:, 6:], torch.zeros_like(grad[:, 6:])), kind
