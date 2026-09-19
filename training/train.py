import math
import time

import torch
import torch.nn as nn

from data.dataset import create_batch
from transformer.transformer_lm import TransformerLM


# ============================================================
# Configuration
# ============================================================

TRAIN_DATA_PATH = "data/train_tokens.pt"
VAL_DATA_PATH = "data/val_tokens.pt"

VOCAB_SIZE = 1000

D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
D_FF = 512

NUM_EXPERTS = 4
TOP_K = 2

BLOCK_SIZE = 128
BATCH_SIZE = 8

LEARNING_RATE = 3e-4
NUM_STEPS = 10

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load pre-tokenized dataset
# ============================================================

train_data = torch.load(TRAIN_DATA_PATH)
val_data = torch.load(VAL_DATA_PATH)

vocab_size = VOCAB_SIZE


print("=" * 60)
print("DATASET")
print("=" * 60)

print("Vocabulary size  :", vocab_size)
print("Train tokens     :", len(train_data))
print("Validation tokens:", len(val_data))
print("Device           :", DEVICE)


# ============================================================
# Move data to device
# ============================================================

train_data = train_data.to(DEVICE)
val_data = val_data.to(DEVICE)


# ============================================================
# Create model
# ============================================================

model = TransformerLM(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
)

model = model.to(DEVICE)


# ============================================================
# Loss + optimizer
# ============================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Parameter count
# ============================================================

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Parameters       :", num_parameters)


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data,
    batch_size,
    block_size,
):
    model.eval()

    inputs, targets = create_batch(
        data,
        batch_size=batch_size,
        block_size=block_size,
    )

    logits = model(inputs)

    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

    model.train()

    return loss.item()


# ============================================================
# Training
# ============================================================

model.train()

start_time = time.time()

for step in range(NUM_STEPS):

    # --------------------------------------------------------
    # Get training batch
    # --------------------------------------------------------

    inputs, targets = create_batch(
        train_data,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    optimizer.zero_grad()

    logits = model(inputs)

    # logits:
    # (B, T, vocab_size)

    # targets:
    # (B, T)

    # --------------------------------------------------------
    # Cross-entropy loss
    # --------------------------------------------------------

    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # AdamW parameter update
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if step % 1 == 0:

        train_loss = loss.item()

        val_loss = evaluate(
            model,
            val_data,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
        )

        # ----------------------------------------------------
        # Perplexity
        # ----------------------------------------------------

        val_perplexity = math.exp(val_loss)

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        elapsed = time.time() - start_time

        progress = (
            (step + 1) / NUM_STEPS
        ) * 100

        steps_per_second = (
            (step + 1) / elapsed
        )

        remaining_steps = (
            NUM_STEPS - (step + 1)
        )

        remaining_seconds = (
            remaining_steps / steps_per_second
        )

        remaining_minutes = (
            remaining_seconds / 60
        )

        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

        if DEVICE == "cuda":

            gpu_memory = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            gpu_memory_text = (
                f"{gpu_memory:.2f} GB"
            )

        else:

            gpu_memory_text = "N/A"

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print(
            f"Step {step + 1:4d}/{NUM_STEPS} "
            f"({progress:5.1f}%) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PPL: {val_perplexity:.2f} | "
            f"Speed: {steps_per_second:.2f} step/s | "
            f"ETA: {remaining_minutes:.1f} min | "
            f"GPU: {gpu_memory_text}"
        )