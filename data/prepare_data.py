import torch

from data.tokenizer import CharacterTokenizer


# -----------------------------------------
# Load text
# -----------------------------------------

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()


# -----------------------------------------
# Create tokenizer
# -----------------------------------------

tokenizer = CharacterTokenizer(text)


# -----------------------------------------
# Encode entire dataset
# -----------------------------------------

data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)


# -----------------------------------------
# Print information
# -----------------------------------------

print("Text length :", len(text))
print("Vocab size  :", tokenizer.vocab_size)
print("Data shape  :", data.shape)

print("First 20 IDs:")
print(data[:20])

print("Decoded:")
print(tokenizer.decode(data[:20].tolist()))