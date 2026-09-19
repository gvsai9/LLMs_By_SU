from tokenizer import Tokenizer
from dataset import load_tokenized_data


tokenizer = Tokenizer.from_files(
    vocab_filepath="data/vocab.json",
    merges_filepath="data/merges.txt",
    special_tokens=[]
)

train_data, val_data = load_tokenized_data(
    text_path="data/train.txt",
    tokenizer=tokenizer
)

print("Train tokens:", train_data.shape)
print("Validation tokens:", val_data.shape)

print("First 20 token IDs:")
print(train_data[:20])

print("Decoded:")
print(
    tokenizer.decode(
        train_data[:20].tolist()
    )
)