class CharacterTokenizer:

    def __init__(self, text):

        # Create vocabulary
        self.chars = sorted(set(text))

        # Character → ID
        self.stoi = {
            ch: i
            for i, ch in enumerate(self.chars)
        }

        # ID → Character
        self.itos = {
            i: ch
            for i, ch in enumerate(self.chars)
        }

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):

        return [
            self.stoi[ch]
            for ch in text
        ]

    def decode(self, ids):

        return "".join(
            self.itos[i]
            for i in ids
        )