import torch

from tokenizer.tokenizer import Tokenizer


TEXT_PATH = "data/train.txt"
VOCAB_PATH = "data/vocab.json"
MERGES_PATH = "data/merges.txt"

TRAIN_OUTPUT_PATH = "data/train_tokens.pt"
VAL_OUTPUT_PATH = "data/val_tokens.pt"

TRAIN_RATIO = 0.9


if __name__ == "__main__":

    tokenizer = Tokenizer.from_files(
        vocab_filepath=VOCAB_PATH,
        merges_filepath=MERGES_PATH,
        special_tokens=[],
    )

    print("Loading dataset...")

    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing dataset...")

    token_ids = tokenizer.encode(text)

    data = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    split = int(TRAIN_RATIO * len(data))

    train_data = data[:split]
    val_data = data[split:]

    torch.save(train_data, TRAIN_OUTPUT_PATH)
    torch.save(val_data, VAL_OUTPUT_PATH)

    print("=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print("Vocabulary size  :", len(tokenizer.vocab))
    print("Total tokens     :", len(data))
    print("Train tokens     :", len(train_data))
    print("Validation tokens:", len(val_data))
    print("Saved train      :", TRAIN_OUTPUT_PATH)
    print("Saved validation :", VAL_OUTPUT_PATH)