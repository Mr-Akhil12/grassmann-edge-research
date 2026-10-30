# Original-paper interpretation and implementation audit

Paper: Zhang Chong, "Attention Is Not What You Need", arXiv:2512.19428.
https://arxiv.org/html/2512.19428v1

Reproductions worth examining:
https://github.com/Infatoshi/grassmann-flows
https://github.com/kanenorman/grassmann

## Causal alignment question
The manuscript describes pairing position t with t+delta while presenting a
causal left-to-right language-modeling architecture. Under standard
next-token prediction and unshifted output alignment, this is future
information leakage. We cannot assume a true paper bug until verifying
intended indexing and each experimental model's code, but any independently
tested autoregressive version MUST explicitly pass prefix invariance and
future-to-past gradient tests.

This prototype uses only t and t-delta for actual modeling; its intentionally
future-looking setting exists exclusively for a negative causality test.

## Ambiguities for a faithful reproduction
- Whether several offset features are added, concatenated or normalized.
- Normalization placement relative to geometric aggregation and projection.
- Outer residual and FFN placement.
- Tokenization, train/evaluation split and optimizer details.
- Rank, offset schedule and per-layer receptive-field growth.
- Causal masking and whether labels are shifted during training.
- Parameter count and context-dependent running cost.
- Difference between theoretical asymptotics and actual optimized kernels.

## Interpretation discipline
Small classification accuracy differences do not establish strong NLP
advantages. Linear theoretical complexity does not guarantee faster Android
inference. A useful positive result must compare quality at a matched cost,
against local mixing as well as full attention. Disclose code variations.
