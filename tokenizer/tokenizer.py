import regex
from collections import Counter
from typing import Iterable, Iterator


GPT2_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        # --------------------------------------------------
        # Add special tokens to vocabulary
        # --------------------------------------------------
        self.special_token_to_id = {}

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes in self.vocab.values():
                token_id = next(
                    i for i, b in self.vocab.items()
                    if b == token_bytes
                )
            else:
                token_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[token_id] = token_bytes

            self.special_token_to_id[token] = token_id

        # --------------------------------------------------
        # bytes -> ID
        # --------------------------------------------------
        self.bytes_to_id = {
            b: i for i, b in self.vocab.items()
        }

        # --------------------------------------------------
        # Merge -> resulting token ID
        # --------------------------------------------------
        self.merge_to_id = {}

        for left, right in self.merges:
            merged = left + right

            if merged in self.bytes_to_id:
                self.merge_to_id[(left, right)] = (
                    self.bytes_to_id[merged]
                )

        # --------------------------------------------------
        # Merge priority
        # --------------------------------------------------
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens=None,
    ):
        import json

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {
            int(token_id): bytes.fromhex(token_bytes)
            for token_id, token_bytes in raw_vocab.items()
        }

        merges = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                left, right = line.rstrip("\n").split("\t")
                merges.append(
                    (
                        bytes.fromhex(left),
                        bytes.fromhex(right),
                    )
                )

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    # ======================================================
    # Encode one pre-tokenized piece
    # ======================================================

    def _encode_piece(self, text):
        token_ids = list(text.encode("utf-8"))

        while len(token_ids) >= 2:

            best_pair = None
            best_rank = float("inf")

            for i in range(len(token_ids) - 1):
                pair = (
                    self.vocab[token_ids[i]],
                    self.vocab[token_ids[i + 1]],
                )

                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_id = self.merge_to_id[best_pair]

            new_tokens = []
            i = 0

            while i < len(token_ids):

                if (
                    i + 1 < len(token_ids)
                    and (
                        self.vocab[token_ids[i]],
                        self.vocab[token_ids[i + 1]],
                    ) == best_pair
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(token_ids[i])
                    i += 1

            token_ids = new_tokens

        return token_ids

    # ======================================================
    # Encode
    # ======================================================

    def encode(self, text):
        if not self.special_tokens:
            pieces = regex.findall(GPT2_REGEX, text)

            result = []

            for piece in pieces:
                result.extend(self._encode_piece(piece))

            return result

        # Longest special tokens first
        specials = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        pattern = "|".join(
            regex.escape(token)
            for token in specials
        )

        parts = regex.split(
            f"({pattern})",
            text,
        )

        result = []

        for part in parts:
            if not part:
                continue

            if part in self.special_token_to_id:
                result.append(
                    self.special_token_to_id[part]
                )
            else:
                pieces = regex.findall(
                    GPT2_REGEX,
                    part,
                )

                for piece in pieces:
                    result.extend(
                        self._encode_piece(piece)
                    )

        return result

    # ======================================================
    # Lazy iterable encoding
    # ======================================================

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:

        for text in iterable:
            yield from self.encode(text)

    # ======================================================
    # Decode
    # ======================================================

    def decode(self, ids):
        byte_string = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return byte_string.decode(
            "utf-8",
            errors="replace",
        )