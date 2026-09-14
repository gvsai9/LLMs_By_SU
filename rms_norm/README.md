
---

# `rmsnorm/README.md`

```markdown
# RMSNorm

## Purpose

RMSNorm normalizes the magnitude of activations flowing through the Transformer.

For an input vector `x`:

\[
RMS(x)
=
\sqrt{
\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon
}
\]

The normalized vector is:

\[
\hat{x} = \frac{x}{RMS(x)}
\]

A learnable scale vector is then applied:

\[
y = g \odot \hat{x}
\]

Therefore:

\[
\boxed{
y =
\frac{x}
{\sqrt{\operatorname{mean}(x^2)+\epsilon}}
\odot g
}
\]

## Learnable Parameter

RMSNorm has one learnable vector:

\[
g \in \mathbb{R}^{d}
\]

If:

```text
d = 768