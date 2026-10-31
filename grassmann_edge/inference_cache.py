"""Incremental inference for the existing TinyLM Transformer/local controls.

A transparent reference, NOT optimized production KV caching. Transformer
stores all K/V by layer (O(L*d) state); local mixer stores last max(offset)
reduced vectors (O(max(offset)*rank) per layer). No dropout in eval mode.
"""
import torch
from torch.nn import functional as F
from grassmann_edge.models import TinyLM, TransformerMix, LocalMix


@torch.no_grad()
def incremental_tiny(model: TinyLM, ids: torch.Tensor):
    if model.kind not in ("transformer", "local"):
        raise ValueError("Use PaperGrassmannLM for Grassmann cached decode")
    if model.training:
        raise ValueError("Eval mode required for cache parity")
    if ids.ndim != 2 or not 1 <= ids.shape[1] <= model.context:
        raise ValueError("Invalid context")
    caches = [None] * len(model.blocks)
    logits = []
    for t in range(ids.shape[1]):
        x = model.embed(ids[:, t:t + 1]) + model.pos(
            torch.tensor([t], device=ids.device))
        for layer_idx, block in enumerate(model.blocks):
            if isinstance(block, TransformerMix):
                b, _, d = x.shape
                qkv = block.qkv(x).reshape(b, 1, 3, block.heads, d // block.heads)
                q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
                if caches[layer_idx] is not None:
                    prior_k, prior_v = caches[layer_idx]
                    k = torch.cat((prior_k, k), dim=2)
                    v = torch.cat((prior_v, v), dim=2)
                caches[layer_idx] = (k, v)
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                h = block.mix_norm(x + block.dropout(
                    block.out(attn.transpose(1, 2).reshape(b, 1, d))))
                x = block.ff_norm(h + block.dropout(block.ff(h)))
            elif isinstance(block, LocalMix):
                current = block.reduce(x)
                prev = caches[layer_idx] if caches[layer_idx] is not None else []
                g = torch.zeros_like(x)
                count = 0
                for offset in block.offsets:
                    if len(prev) >= offset:
                        g += block.project(torch.cat((current, prev[-offset]), -1))
                        count += 1
                if count:
                    g /= count
                keep = max(block.offsets)
                caches[layer_idx] = (prev + [current])[-keep:]
                alpha = torch.sigmoid(block.gate(torch.cat((x, g), -1)))
                h = block.mix_norm(block.dropout(alpha * x + (1 - alpha) * g))
                x = block.ff_norm(h + block.dropout(block.ff(h)))
            else:
                raise TypeError(type(block))
        logits.append(model.head(model.final_norm(x)))
    return torch.cat(logits, dim=1)


def model_state_bytes(model: TinyLM, context: int, batch: int = 1) -> int:
    """Approximate *tensor payload only*, not CUDA allocated memory or allocator."""
    element = next(model.parameters()).element_size()
    if model.kind == "transformer":
        return 2 * len(model.blocks) * batch * context * model.embed.embedding_dim * element
    if model.kind == "local":
        return sum(batch * min(context, max(b.offsets)) *
                   b.reduce.out_features * element for b in model.blocks)
    raise ValueError("Unsupported model kind")
