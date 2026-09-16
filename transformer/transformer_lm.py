import torch
import torch.nn as nn

from embedding_layer.embeddinglayer import EmbeddingLayer
from rms_norm.rmsnorm import RMSNorm
from transformer.transformer_block import TransformerBlock


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        num_experts,
        top_k
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            d_model=d_model
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(d_model)

        # Language-model head
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids):

        # Token IDs → embeddings
        x = self.embedding(input_ids)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # d_model → vocabulary
        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":

    vocab_size = 100

    d_model = 8
    num_layers = 2
    num_heads = 2
    d_ff = 32

    num_experts = 4
    top_k = 2

    B = 2
    T = 5

    input_ids = torch.randint(
        0,
        vocab_size,
        (B, T)
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k
    )

    logits = model(input_ids)

    print("Input IDs :", input_ids.shape)
    print("Logits    :", logits.shape)