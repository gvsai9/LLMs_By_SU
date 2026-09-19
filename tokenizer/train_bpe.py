import regex
from collections import Counter
from tokenizer.tokenizer import GPT2_REGEX
def train_bpe(
    input_path,
    vocab_size,
    special_tokens,
):
    # --------------------------------------------------
    # Read corpus
    # --------------------------------------------------
    print("Opening input file:", input_path)
    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read()

    # --------------------------------------------------
    # Protect special tokens
    # --------------------------------------------------
    print("Protecting special tokens:", special_tokens)
    special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    print("Splitting text into chunks...")
    if special_tokens:
        special_pattern = "|".join(
            regex.escape(token)
            for token in special_tokens
        )

        chunks = regex.split(
            f"(?:{special_pattern})",
            text,
        )
    else:
        chunks = [text]

    # --------------------------------------------------
    # Pre-tokenize
    # --------------------------------------------------

    pre_tokens = []
    print("Pre-tokenizing text...")
    for chunk in chunks:
        pre_tokens.extend(
            regex.findall(
                GPT2_REGEX,
                chunk,
            )
        )

    # --------------------------------------------------
    # Convert to byte sequences
    # --------------------------------------------------
    print("Converting pre-tokens to byte sequences...")
    sequences = [
        list(piece.encode("utf-8"))
        for piece in pre_tokens
    ]

    # --------------------------------------------------
    # Initial vocabulary
    # --------------------------------------------------
    print("Creating initial vocabulary...")
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    merges = []
    print("Initial vocabulary size:", len(vocab))
    # Number of normal tokens we need
    target_normal_vocab = (
        vocab_size - len(special_tokens)
    )

    # --------------------------------------------------
    # Learn merges
    # --------------------------------------------------
    print("Learning merges...")
    while len(vocab) < target_normal_vocab:

        pair_counts = Counter()

        for sequence in sequences:
            for i in range(len(sequence) - 1):
                pair_counts[
                    (
                        sequence[i],
                        sequence[i + 1],
                    )
                ] += 1

        if not pair_counts:
            break
        print("Current vocabulary size:", len(vocab))
        # Most frequent pair
        # Lexicographically larger pair wins ties
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        left, right = best_pair

        new_id = len(vocab)

        # Save merge as byte strings
        merges.append(
            (
                vocab[left],
                vocab[right],
            )
        )

        # Create merged vocabulary entry
        vocab[new_id] = (
            vocab[left] + vocab[right]
        )

        # --------------------------------------------------
        # Replace occurrences
        # --------------------------------------------------

        new_sequences = []

        for sequence in sequences:

            new_sequence = []

            i = 0

            while i < len(sequence):

                if (
                    i + 1 < len(sequence)
                    and sequence[i] == left
                    and sequence[i + 1] == right
                ):
                    new_sequence.append(new_id)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1

            new_sequences.append(new_sequence)

        sequences = new_sequences

    # --------------------------------------------------
    # Add special tokens
    # --------------------------------------------------
    print("Adding special tokens to vocabulary...")
    for token in special_tokens:
        token_id = len(vocab)

        vocab[token_id] = token.encode("utf-8")

    return vocab, merges

from pathlib import Path
import json


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    INPUT_PATH = DATA_DIR / "train.txt"
    VOCAB_PATH = DATA_DIR / "vocab.json"
    MERGES_PATH = DATA_DIR / "merges.txt"

    VOCAB_SIZE = 1000
    SPECIAL_TOKENS = []

    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    # --------------------------------------------------------
    # Save vocabulary
    # --------------------------------------------------------
    print("Saving vocabulary to:", VOCAB_PATH)
    with open(
        VOCAB_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                str(token_id): token_bytes.hex()
                for token_id, token_bytes in vocab.items()
            },
            f,
            indent=2,
        )
    print("Vocabulary saved to:", VOCAB_PATH)
    # --------------------------------------------------------
    # Save merges
    # --------------------------------------------------------

    with open(
        MERGES_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for left, right in merges:
            f.write(
                f"{left.hex()}\t{right.hex()}\n"
            )

    print("=" * 60)
    print("BPE TRAINING COMPLETE")
    print("=" * 60)
    print("Vocabulary size :", len(vocab))
    print("Number of merges:", len(merges))
    print("Saved vocabulary:", VOCAB_PATH)
    print("Saved merges     :", MERGES_PATH)