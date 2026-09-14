import torch
import torch.nn as nn
from linear_layer.linearlayer import LinearLayer

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = LinearLayer(d_model, d_ff)
        self.w3 = LinearLayer(d_model, d_ff)
        self.w2 = LinearLayer(d_ff, d_model)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.w1(x))
        value = self.w3(x)

        x = gate * value

        return self.w2(x)
if __name__ == "__main__":
    layer = SwiGLUFFN(4, 8)

    x = torch.randn(2, 3, 4)
    y = layer(x)

    print(x.shape)
    print(y.shape)