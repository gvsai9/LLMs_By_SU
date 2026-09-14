import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )

        return self.weight * (x / rms)
if __name__ == "__main__":
    layer = RMSNorm(4)

    x = torch.randn(2, 3, 4)
    y = layer(x)

    print(x.shape)
    print(y.shape)
    print(layer.weight.shape)