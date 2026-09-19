import torch


def create_batch(data, batch_size, block_size):

    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    inputs = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    targets = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return inputs, targets


def load_tokenized_data(
    text_path,
    tokenizer,
    train_ratio=0.9,
):
    with open(
        text_path,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # Use your completed BPE tokenizer
    token_ids = tokenizer.encode(text)

    data = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    split = int(
        train_ratio * len(data)
    )

    train_data = data[:split]
    val_data = data[split:]

    return train_data, val_data