import regex
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
        # Special token -> ID
        # --------------------------------------------------
        self.special_token_to_id = {}

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")

            if special_bytes in self.vocab.values():
                token_id = next(
                    token_id
                    for token_id, token_bytes in self.vocab.items()
                    if token_bytes == special_bytes
                )
            else:
                token_id = max(self.vocab.keys()) + 1
                self.vocab[token_id] = special_bytes

            self.special_token_to_id[special_token] = token_id

        # --------------------------------------------------
        # bytes -> token ID
        # --------------------------------------------------
        self.bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # --------------------------------------------------
        # Merge -> token ID
        # --------------------------------------------------
        self.merge_to_id = {}

        for left, right in self.merges:
            merged_bytes = left + right

            if merged_bytes in self.bytes_to_id:
                self.merge_to_id[(left, right)] = (
                    self.bytes_to_id[merged_bytes]
                )

        # --------------------------------------------------
        # Merge priority
        # --------------------------------------------------
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    # ======================================================
    # Encode one normal piece
    # ======================================================

    def _encode_piece(self, text):

      # Start with individual byte IDs
      tokens = list(text.encode("utf-8"))

      while len(tokens) > 1:

          best_pair = None
          best_rank = None

          # Find highest-priority applicable merge
          for i in range(len(tokens) - 1):

              pair = (
                  self.vocab[tokens[i]],
                  self.vocab[tokens[i + 1]]
              )

              if pair in self.merge_ranks:

                  rank = self.merge_ranks[pair]

                  if best_rank is None or rank < best_rank:
                      best_rank = rank
                      best_pair = pair

          # No more merges
          if best_pair is None:
              break

          new_id = self.merge_to_id[best_pair]

          new_tokens = []
          i = 0

          while i < len(tokens):

              if i + 1 < len(tokens):

                  pair = (
                      self.vocab[tokens[i]],
                      self.vocab[tokens[i + 1]]
                  )

                  if pair == best_pair:
                      new_tokens.append(new_id)
                      i += 2
                      continue

              new_tokens.append(tokens[i])
              i += 1

          tokens = new_tokens

      return tokens
    # ======================================================
    # Encode
    # ======================================================

    def encode(self, text: str) -> list[int]:

        output_ids = []

        # --------------------------------------------------
        # No special tokens
        # --------------------------------------------------

        if not self.special_tokens:

            pieces = regex.findall(
                GPT2_REGEX,
                text
            )

            for piece in pieces:
                output_ids.extend(
                    self._encode_piece(piece)
                )

            return output_ids

        # --------------------------------------------------
        # Special tokens
        #
        # Sort longest first so overlapping special tokens
        # are handled correctly.
        # --------------------------------------------------

        sorted_special_tokens = sorted(
            self.special_tokens,
            key=len,
            reverse=True
        )

        special_pattern = "|".join(
            regex.escape(token)
            for token in sorted_special_tokens
        )

        parts = regex.split(
            f"({special_pattern})",
            text
        )

        for part in parts:

            if not part:
                continue

            # Special token
            if part in self.special_token_to_id:

                output_ids.append(
                    self.special_token_to_id[part]
                )

            # Normal text
            else:

                pieces = regex.findall(
                    GPT2_REGEX,
                    part
                )

                for piece in pieces:

                    output_ids.extend(
                        self._encode_piece(piece)
                    )

        return output_ids

    # ======================================================
    # Encode iterable
    # ======================================================

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:

        for text in iterable:

            for token_id in self.encode(text):
                yield token_id

    # ======================================================
    # Decode
    # ======================================================

    def decode(self, ids: list[int]) -> str:

        byte_string = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return byte_string.decode(
            "utf-8",
            errors="replace"
        )