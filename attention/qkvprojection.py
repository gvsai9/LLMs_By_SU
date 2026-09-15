import torch
import torch.nn as nn

from linear_layer.linearlayer import LinearLayer


class QKVProjection(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()

        self.w_q = LinearLayer(d_model, d_head)
        self.w_k = LinearLayer(d_model, d_head)
        self.w_v = LinearLayer(d_model, d_head)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        return q, k, v
if __name__ == "__main__":
    d_model = 512
    d_head = 64
    batch_size = 2
    seq_length = 10

    qkv_projection = QKVProjection(d_model, d_head)

    x = torch.randn(batch_size, seq_length, d_model)

    q, k, v = qkv_projection(x)

    print("Q shape:", q.shape)
    print("K shape:", k.shape)
    print("V shape:", v.shape)
        