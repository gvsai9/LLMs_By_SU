import torch


def scaled_dot_product_attention_scores(q, k):
    d_k = q.shape[-1]

    scores = q @ k.transpose(-2, -1)

    scores = scores / torch.sqrt(
        torch.tensor(
            d_k,
            dtype=q.dtype,
            device=q.device
        )
    )

    return scores


def apply_causal_mask(scores):
    T = scores.shape[-1]

    mask = torch.triu(
        torch.ones(
            T,
            T,
            device=scores.device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    return scores.masked_fill(
        mask,
        float("-inf")
    )


def attention_softmax(scores):
    return torch.softmax(scores, dim=-1)

def attention_output(attention_weights, v):
    return attention_weights @ v

def scaled_dot_product_attention(q, k, v):
    scores = scaled_dot_product_attention_scores(q, k)

    scores = apply_causal_mask(scores)

    attention_weights = torch.softmax(
        scores,
        dim=-1
    )

    output = attention_weights @ v

    return output

if __name__ == "__main__":
    q = torch.randn(2, 5, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 4)

    output = scaled_dot_product_attention(q, k, v)

    print("Q:", q.shape)
    print("K:", k.shape)
    print("V:", v.shape)
    print("Output:", output.shape)