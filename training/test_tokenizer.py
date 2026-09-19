from tokenizer.tokenizer import Tokenizer


VOCAB_PATH = "data/vocab.json"
MERGES_PATH = "data/merges.txt"


tokenizer = Tokenizer.from_files(
    vocab_filepath=VOCAB_PATH,
    merges_filepath=MERGES_PATH,
    special_tokens=[],
)


text = "The transformer model learns from data."


ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)


print("=" * 60)
print("TOKENIZER TEST")
print("=" * 60)

print("Original:")
print(text)

print("\nToken IDs:")
print(ids)

print("\nNumber of tokens:")
print(len(ids))

print("\nDecoded:")
print(decoded)

print("\nExact match:")
print(text == decoded)

print("\nVocabulary size:")
print(len(tokenizer.vocab))