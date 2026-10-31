# P0: Literal paper pairing vs autoregressive causality

**Status: candidate interpretation / falsifiable code test, not author's intent resolved.**

Original: Zhang Chong, *Attention Is Not What You Need*, arXiv:2512.19428v1,
https://arxiv.org/html/2512.19428v1#S3.SS2 (Section 3.2).

## Literal equations and index conflict

The paper explicitly defines reduced token vectors `z_t = W_red h_t + b_red`,
then pairs `(z_t, z_(t+Delta))` for positive offsets. Its next sentence calls
this direction "causal" for left-to-right language modeling. Under standard
next-token prediction with output position t predicting token t+1, this
right-looking definition cannot be causally masked *as written*. The claim is
a contradiction in the stated indexing conventions, not proof of a bug in
unseen training code or that the author intended this exact implementation.

For rank r, coordinates are `p_(i,j) = z_(t,i) z_(neighbor,j) -
z_(t,j) z_(neighbor,i)`, for `1 <= i < j <= r` (r choose 2 values).
Normalize the vector by `max(l2_norm, eps)`. Project back to d, average
valid offsets, compute sigmoid gate on concatenated h and g, mix
`alpha * h + (1-alpha) * g`, apply LayerNorm then dropout,
and compute LayerNorm(mixed + FFN(mixed)).

## Implemented variants

- `paper_literal`: right-looking t+Delta as written (not usable for causal
  autoregressive decoding). Included for falsification only.
- `past_causal`: replace t+Delta with t-Delta, preserving geometric
  coordinate order and Section 3.2 downstream computations.
- Boundary policy for positions with zero eligible neighbors: `g_t=0`.
  The paper does not specify this policy. Any later experiments must disclose it.
- Normalization enabled in both variants (paper describes it as optional).
- No claim that other unspecified implementation details were recovered.

## Tests and falsification criteria

- Compare vectorized Pluecker coordinates and gradients against independent
  explicit double-precision scalar-loop oracle.
- Mutate future token states and ensure earlier outputs are invariant in
  `past_causal` but change under `paper_literal`.
- Verify gradients of earlier outputs with respect to future inputs are zero
  for causal model and generally nonzero for paper-literal model.
- Check short sequences, boundary normalization and numeric stability.
- Verify an incremental causal reference, with low-rank state per layer,
  matches full-sequence outputs within declared fp32 tolerance.

The incremental reference is an algorithmic parity test, **not** an optimized
edge kernel nor measurement of generative AI quality. Caching retains
max(offset) reduced vectors per layer, thus memory bound with respect to
generated sequence length for fixed offsets, *excluding embeddings and
output logits*. Learned absolute position embeddings still limit context.

## External validity and remaining open questions

- Whether the original experiments actually used the written future pairing
  is unknown without definitive author source/code or clarification.
- The pairing's causal correction may change original reported quality.
- This is not equivalent to reproducing WikiText-2 perplexity or the
  original 13–18 million parameter runs.
- Layer receptive-field maximum (inclusive): `1 + layers * max(offsets)`
  when every layer uses the same offsets. Finite receptive field is a
  substantive limitation compared with unrestricted causal attention.
- Time complexity is linear in sequence length at fixed rank and offsets,
  but kernel count, memory traffic and dispatch overhead may dominate.
