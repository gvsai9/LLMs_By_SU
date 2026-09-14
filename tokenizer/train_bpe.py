import regex
from collections import Counter


def train_bpe(
    input_path,
    vocab_size,
    special_tokens,
):
    # --------------------------------------------------
    # Read corpus
    # --------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read()

    # --------------------------------------------------
    # Protect special tokens
    # --------------------------------------------------

    special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

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

    sequences = [
        list(piece.encode("utf-8"))
        for piece in pre_tokens
    ]

    # --------------------------------------------------
    # Initial vocabulary
    # --------------------------------------------------

    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    merges = []

    # Number of normal tokens we need
    target_normal_vocab = (
        vocab_size - len(special_tokens)
    )

    # --------------------------------------------------
    # Learn merges
    # --------------------------------------------------

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

    for token in special_tokens:
        token_id = len(vocab)

        vocab[token_id] = token.encode("utf-8")

    return vocab, merges