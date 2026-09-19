import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = nn.Parameter(
            torch.randn(output_dim, input_dim)
        )

    def forward(self, x):
        return x @ self.weights.T
if __name__ == "__main__":
    layer = LinearLayer(3, 2)
    x = torch.randn(4, 3)

    y = layer(x)

    print(layer.weights.shape)
    print(x.shape)
    print(y.shape)
    print(list(layer.parameters()))