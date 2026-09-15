import torch
import torch.nn as nn

from linear_layer.linearlayer import LinearLayer
from attention.sdp import scaled_dot_product_attention
from attention.rope import apply_rope

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V projections
        self.w_q = LinearLayer(d_model, d_model)
        self.w_k = LinearLayer(d_model, d_model)
        self.w_v = LinearLayer(d_model, d_model)

        # Final output projection
        self.w_o = LinearLayer(d_model, d_model)

    def forward(self, x):

        # x: (B, T, d_model)
        B, T, _ = x.shape

        # ------------------------------------------------
        # 1. Create Q, K, V
        # ------------------------------------------------

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # q, k, v:
        # (B, T, d_model)

        # ------------------------------------------------
        # 2. Split d_model into multiple heads
        # ------------------------------------------------

        q = q.view(
            B, T,
            self.num_heads,
            self.d_head
        )

        k = k.view(
            B, T,
            self.num_heads,
            self.d_head
        )

        v = v.view(
            B, T,
            self.num_heads,
            self.d_head
        )

        # ------------------------------------------------
        # 3. Put heads before tokens
        # ------------------------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Now:
        # (B, num_heads, T, d_head)
        # Apply positional information to Q and K
        q = apply_rope(q)
        k = apply_rope(k)
        # ------------------------------------------------
        # 4. Attention for every head
        # ------------------------------------------------

        out = scaled_dot_product_attention(
            q, k, v
        )

        # out:
        # (B, num_heads, T, d_head)

        # ------------------------------------------------
        # 5. Put tokens before heads
        # ------------------------------------------------

        out = out.transpose(1, 2)

        # (B, T, num_heads, d_head)

        # ------------------------------------------------
        # 6. Combine heads
        # ------------------------------------------------

        out = out.contiguous().view(
            B,
            T,
            self.d_model
        )

        # (B, T, d_model)

        # ------------------------------------------------
        # 7. Final output projection
        # ------------------------------------------------

        out = self.w_o(out)

        return out


if __name__ == "__main__":

    x = torch.randn(2, 5, 8)

    mha = MultiHeadAttention(
        d_model=8,
        num_heads=2
    )

    output = mha(x)

    print("Input :", x.shape)
    print("Output:", output.shape)