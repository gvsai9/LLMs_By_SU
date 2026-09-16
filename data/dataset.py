import torch


def create_batch(data, batch_size, block_size):

    # Random starting positions
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    # Input sequences
    inputs = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    # Next-token targets
    targets = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return inputs, targets