import torch

from data.tokenizer import CharacterTokenizer
from data.dataset import create_batch


# -----------------------------------------
# Load text
# -----------------------------------------

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()


# -----------------------------------------
# Tokenizer
# -----------------------------------------

tokenizer = CharacterTokenizer(text)

data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)


# -----------------------------------------
# Create batch
# -----------------------------------------

batch_size = 2
block_size = 8

inputs, targets = create_batch(
    data,
    batch_size=batch_size,
    block_size=block_size
)


# -----------------------------------------
# Display
# -----------------------------------------

print("Inputs shape :", inputs.shape)
print("Targets shape:", targets.shape)

print("\nInput IDs:")
print(inputs)

print("\nTarget IDs:")
print(targets)

print("\nInput text:")
print([
    tokenizer.decode(row.tolist())
    for row in inputs
])

print("\nTarget text:")
print([
    tokenizer.decode(row.tolist())
    for row in targets
])