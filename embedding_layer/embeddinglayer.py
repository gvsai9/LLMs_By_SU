import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(vocab_size, embedding_dim)
        )

    def forward(self, x):
        return self.weight[x]
if __name__ == "__main__":
    layer = EmbeddingLayer(10, 4)

    x = torch.tensor([2, 7, 4])
    y = layer(x)

    print(layer.weight.shape)
    print(x.shape)
    print(y.shape)