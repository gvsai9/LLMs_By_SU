from collections import Counter
from common import pretokenize


def train_bpe(text, vocab_size):
    # 1. Pre-tokenize
    pre_tokens = pretokenize(text)

    # 2. Convert each pre-token to UTF-8 bytes
    tokens = [
        list(token.encode("utf-8"))
        for token in pre_tokens
    ]

    # 3. Initial vocabulary: 256 byte tokens
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    # 4. Store merges in order
    merges = []

    # 5. Initial pair counts
    pair_counts = Counter()

    for token in tokens:
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_counts[pair] += 1

    # 6. Keep learning merges
    while len(vocab) < vocab_size:

        if not pair_counts:
            break

        # Most frequent pair
        # Tie → lexicographically greater pair
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], p)
        )

        left, right = best_pair

        # New token ID
        new_id = len(vocab)

        # Record merge
        merges.append(
            (vocab[left], vocab[right])
        )

        # Create vocabulary entry
        vocab[new_id] = vocab[left] + vocab[right]

        # 7. Replace pair and update counts
        new_tokens = []

        for token in tokens:

            new_token = []
            i = 0

            while i < len(token):

                if (
                    i + 1 < len(token)
                    and token[i] == left
                    and token[i + 1] == right
                ):

                    # Remove old neighboring pairs
                    if i > 0:
                        pair_counts[
                            (token[i - 1], left)
                        ] -= 1

                    if i + 2 < len(token):
                        pair_counts[
                            (right, token[i + 2])
                        ] -= 1

                    # Remove merged pair
                    pair_counts[best_pair] -= 1

                    # Add new neighboring pairs
                    if i > 0:
                        pair_counts[
                            (token[i - 1], new_id)
                        ] += 1

                    if i + 2 < len(token):
                        pair_counts[
                            (new_id, token[i + 2])
                        ] += 1

                    new_token.append(new_id)
                    i += 2

                else:
                    new_token.append(token[i])
                    i += 1

            new_tokens.append(new_token)

        tokens = new_tokens

        # Remove zero-count pairs
        pair_counts = Counter({
            pair: count
            for pair, count in pair_counts.items()
            if count > 0
        })

    return vocab, merges