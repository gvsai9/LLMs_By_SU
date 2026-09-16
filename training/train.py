import torch
import torch.nn as nn

from data.tokenizer import CharacterTokenizer
from data.dataset import create_batch
from transformer.transformer_lm import TransformerLM


# -----------------------------------------
# 1. Load training text
# -----------------------------------------

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()


# -----------------------------------------
# 2. Create tokenizer
# -----------------------------------------

tokenizer = CharacterTokenizer(text)

vocab_size = tokenizer.vocab_size

data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)


# -----------------------------------------
# 3. Model configuration
# -----------------------------------------

d_model = 8
num_layers = 2
num_heads = 2
d_ff = 32

num_experts = 4
top_k = 2

block_size = 16
batch_size = 4


# -----------------------------------------
# 4. Create model
# -----------------------------------------

model = TransformerLM(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    num_experts=num_experts,
    top_k=top_k
)


# -----------------------------------------
# 5. Loss + optimizer
# -----------------------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# -----------------------------------------
# 6. Training loop
# -----------------------------------------

num_steps = 200

for step in range(num_steps):

    # Create a fresh random batch
    inputs, targets = create_batch(
        data,
        batch_size=batch_size,
        block_size=block_size
    )

    # Clear old gradients
    optimizer.zero_grad()

    # Forward pass
    logits = model(inputs)

    # Calculate loss
    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print progress
    if step % 20 == 0:
        print(
            f"Step {step:3d} | Loss: {loss.item():.4f}"
        )