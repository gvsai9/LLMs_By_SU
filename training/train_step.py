import torch
import torch.nn as nn

from transformer.transformer_lm import TransformerLM


# -----------------------------------------
# Model configuration
# -----------------------------------------

vocab_size = 100
d_model = 8
num_layers = 2
num_heads = 2
d_ff = 32

num_experts = 4
top_k = 2


# -----------------------------------------
# Create model
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
# Optimizer
# -----------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# -----------------------------------------
# Example token sequence
# -----------------------------------------

input_ids = torch.randint(
    0,
    vocab_size,
    (2, 6)
)


# -----------------------------------------
# Shift input and target
# -----------------------------------------

inputs = input_ids[:, :-1]
targets = input_ids[:, 1:]


# -----------------------------------------
# Loss function
# -----------------------------------------

loss_fn = nn.CrossEntropyLoss()


# -----------------------------------------
# Training loop
# -----------------------------------------

num_steps = 100

for step in range(num_steps):

    # Clear previous gradients
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

    # Print loss
    if step % 10 == 0:
        print(
            f"Step {step:3d} | Loss: {loss.item():.4f}"
        )