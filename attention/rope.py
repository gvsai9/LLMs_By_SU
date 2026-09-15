import torch


def apply_rope(x, theta_base=10000.0):
    """
    Apply Rotary Positional Embedding.

    x shape:
        (B, H, T, d_head)

    d_head must be even.
    """

    B, H, T, d_head = x.shape

    assert d_head % 2 == 0, \
        "d_head must be even for RoPE"

    # --------------------------------------------------
    # Position indices
    # --------------------------------------------------

    positions = torch.arange(
        T,
        device=x.device,
        dtype=x.dtype
    )

    # --------------------------------------------------
    # Frequency for each dimension pair
    # --------------------------------------------------

    dim = torch.arange(
        0,
        d_head,
        2,
        device=x.device,
        dtype=x.dtype
    )

    frequencies = 1.0 / (
        theta_base ** (dim / d_head)
    )

    # --------------------------------------------------
    # Rotation angles
    # --------------------------------------------------

    angles = positions[:, None] * frequencies[None, :]

    # (T, d_head / 2)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # --------------------------------------------------
    # Split x into pairs
    # --------------------------------------------------

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # --------------------------------------------------
    # Rotate
    # --------------------------------------------------

    rotated_even = (
        x_even * cos
        - x_odd * sin
    )

    rotated_odd = (
        x_even * sin
        + x_odd * cos
    )

    # --------------------------------------------------
    # Put dimensions back together
    # --------------------------------------------------

    output = torch.stack(
        [rotated_even, rotated_odd],
        dim=-1
    )

    output = output.flatten(-2)

    return output


if __name__ == "__main__":

    x = torch.randn(
        2, 2, 5, 4
    )

    output = apply_rope(x)

    print("Input :", x.shape)
    print("Output:", output.shape)