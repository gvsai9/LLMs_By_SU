import torch
import torch.nn as nn

from linear_layer.linearlayer import LinearLayer
from attention.sdp import scaled_dot_product_attention
from attention.rope import apply_rope


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # ====================================================
        # Q, K, V projections
        # ====================================================

        self.w_q = LinearLayer(
            d_model,
            d_model
        )

        self.w_k = LinearLayer(
            d_model,
            d_model
        )

        self.w_v = LinearLayer(
            d_model,
            d_model
        )

        # ====================================================
        # Output projection
        # ====================================================

        self.w_o = LinearLayer(
            d_model,
            d_model
        )

    def forward(self, x):

        # x:
        # (B, T, d_model)

        B, T, _ = x.shape

        # ====================================================
        # 1. QKV projections
        # ====================================================

        with torch.profiler.record_function(
            "attention_qkv_projection"
        ):

            q = self.w_q(x)

            k = self.w_k(x)

            v = self.w_v(x)

        # ====================================================
        # 2. Split into heads
        # ====================================================

        q = q.view(
            B,
            T,
            self.num_heads,
            self.d_head
        )

        k = k.view(
            B,
            T,
            self.num_heads,
            self.d_head
        )

        v = v.view(
            B,
            T,
            self.num_heads,
            self.d_head
        )

        # ====================================================
        # 3. Move heads before sequence
        # ====================================================

        q = q.transpose(1, 2)

        k = k.transpose(1, 2)

        v = v.transpose(1, 2)

        # ====================================================
        # 4. RoPE
        # ====================================================

        with torch.profiler.record_function(
            "attention_rope"
        ):

            q = apply_rope(q)

            k = apply_rope(k)

        # ====================================================
        # 5. Scaled Dot Product Attention
        # ====================================================

        with torch.profiler.record_function(
            "attention_sdp"
        ):

            out = scaled_dot_product_attention(
                q,
                k,
                v
            )

        # out:
        # (B, num_heads, T, d_head)

        # ====================================================
        # 6. Move sequence before heads
        # ====================================================

        out = out.transpose(1, 2)

        # ====================================================
        # 7. Combine heads
        # ====================================================

        out = out.contiguous().view(
            B,
            T,
            self.d_model
        )

        # ====================================================
        # 8. Output projection
        # ====================================================

        with torch.profiler.record_function(
            "attention_output_projection"
        ):

            out = self.w_o(out)

        return out


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    x = torch.randn(
        2,
        5,
        8
    )

    mha = MultiHeadAttention(
        d_model=8,
        num_heads=2
    )

    output = mha(x)

    print("Input :", x.shape)
    print("Output:", output.shape)