import torch
import torch.nn as nn

from rms_norm.rmsnorm import RMSNorm
from attention.mha import MultiHeadAttention
from moe.moe import MoEFNN


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_experts,
        top_k
    ):
        super().__init__()

        # -------------------------------
        # Pre-Norm
        # -------------------------------

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # -------------------------------
        # Multi-Head Attention
        # -------------------------------

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        # -------------------------------
        # Mixture-of-Experts FFN
        # -------------------------------

        self.ffn = MoEFNN(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k
        )

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(
            self.norm1(x)
        )

        # MoE FFN + residual
        x = x + self.ffn(
            self.norm2(x)
        )

        return x


if __name__ == "__main__":

    B = 2
    T = 5

    d_model = 8
    num_heads = 2
    d_ff = 32

    num_experts = 4
    top_k = 2

    x = torch.randn(
        B,
        T,
        d_model
    )

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k
    )

    output = block(x)

    print("Input :", x.shape)
    print("Output:", output.shape)