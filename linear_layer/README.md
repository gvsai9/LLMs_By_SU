# Linear Layer

## Purpose

A Linear layer performs a learned linear transformation.

For input dimension `d_in` and output dimension `d_out`:

\[
y = xW^T
\]

where:

\[
W \in \mathbb{R}^{d_{out} \times d_{in}}
\]

## Parameters

The layer contains one learnable parameter:

```python
self.weight