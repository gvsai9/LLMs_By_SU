# Byte Pair Encoding (BPE) Tokenizer

A from-scratch implementation of a **Byte Pair Encoding (BPE) tokenizer**, based on the concepts studied in Stanford CS336 — *Language Models from Scratch*.

This project focuses on understanding how a modern byte-level subword tokenizer works internally, from training the vocabulary and merge rules to encoding text into token IDs and decoding token IDs back into text.

---

## 1. What Is a BPE Tokenizer?

**Byte Pair Encoding (BPE)** is a subword tokenization algorithm used to convert text into a sequence of tokens that can be processed by a language model.

Instead of assigning one token to every complete word, BPE starts with small units and repeatedly merges frequently occurring adjacent units to create larger reusable tokens.

For example, consider the word:

```text
lower
```

Initially, it can be represented using smaller units:

```text
l + o + w + e + r
```

During BPE training, frequent pairs may be learned:

```text
o + w → ow
```

Then:

```text
l + ow → low
```

and potentially:

```text
low + er → lower
```

The exact tokens learned depend on the training corpus and the target vocabulary size.

The fundamental idea is:

- Small units  
- Find frequent adjacent pairs  
- Merge the best pair  
- Create a new token  
- Repeat  

BPE therefore learns a vocabulary of reusable subword tokens.

---

## 2. Why Use Byte-Level BPE?

This implementation uses bytes as the initial units.

A byte can have one of 256 possible values:

```text
0, 1, 2, ..., 255
```

Therefore, the initial vocabulary contains:

- 256 byte tokens  

Instead of starting with a vocabulary containing every possible word, the tokenizer starts from these 256 fundamental byte values and learns larger tokens from them.

The overall representation is:

```text
Unicode Text
      ↓
     UTF-8
      ↓
     Bytes
      ↓
   BPE Tokens
      ↓
   Token IDs
```

Because UTF-8 can represent Unicode text using bytes, byte-level BPE can represent:

- English  
- Other natural languages  
- Unicode characters  
- Emojis  
- Symbols  
- Previously unseen words  

---

## 3. Why Subword Tokenization?

There are several common approaches to tokenization.

### Character-Level Tokenization

Each character becomes a token.

```text
hello
 ↓
h e l l o
```

**Advantages:**

- Very small vocabulary  
- Can represent almost any text  

**Disadvantages:**

- Very long token sequences  
- Language models must process more tokens  

### Word-Level Tokenization

Each word becomes a token.

```text
hello world
 ↓
hello | world
```

**Advantages:**

- Shorter sequences  
- Tokens can represent complete words  

**Disadvantages:**

- Very large vocabulary  
- Unknown words become a problem  
- Different word forms require many vocabulary entries  

### Subword Tokenization

BPE provides a middle ground.

```text
playing
 ↓
play + ing
```

The tokenizer can reuse learned pieces across many words.

Therefore:

- **Character-level** → Small vocabulary, long sequences  
- **Subword-level** → Moderate vocabulary, reusable pieces  
- **Word-level** → Large vocabulary, potential unknown-word problem  

---

## 4. Three Main Phases

The tokenizer can be divided into three major phases:

```text
┌───────────────────────────────────────────────┐
│                 BPE TOKENIZER                 │
├───────────────────────────────────────────────┤
│                                               │
│  1. TRAINING                                  │
│     Learn vocabulary + merge rules            │
│                                               │
│  2. ENCODING                                  │
│     Convert text → token IDs                  │
│                                               │
│  3. DECODING                                  │
│     Convert token IDs → text                  │
│                                               │
└───────────────────────────────────────────────┘
```

These phases have different responsibilities.

- **TRAINING** → Learn how text should be tokenized  
- **ENCODING** → Use the learned tokenizer  
- **DECODING** → Reconstruct text from the token IDs  

---

## 5. Phase 1 — Training

The training phase learns the tokenizer from a training corpus.

The goal is to discover which byte sequences occur frequently enough to become useful tokens.

The training process produces:

- Vocabulary  
- Ordered Merge Rules  

These learned objects are then used during encoding.

---

## 6. Training Workflow

The complete BPE training pipeline is:

```text
Raw Training Corpus
        │
        ▼
GPT-2 Byte-to-Unicode Mapping
        │
        ▼
Pre-tokenization
        │
        ▼
UTF-8 Conversion
        │
        ▼
Initial 256-Byte Vocabulary
        │
        ▼
Count Adjacent Token Pairs
        │
        ▼
Select Best Pair
        │
        ▼
Merge Pair
        │
        ├──────────────► Update Vocabulary
        │
        └──────────────► Record Merge Rule
                              │
                              ▼
                     Update Pair Statistics
                              │
                              ▼
                           Repeat
                              │
                              ▼
                 Target Vocabulary Reached
                              │
                              ▼
              Final Vocabulary + Merge Rules
```

---

## 7. Training Step 1 — GPT-2 Byte-to-Unicode Mapping

The implementation contains:

```python
gpt2_bytes_to_unicode()
```

This function creates a mapping between byte values and Unicode character representations.

Conceptually:

```text
Byte → Unicode representation
```

For example, printable ASCII bytes can be represented directly:

```text
65 → A
97 → a
```

However, not every byte corresponds to a convenient printable character.

For example:

```python
chr(0)
```

represents a null character. The character exists, but it is invisible and inconvenient for displaying token representations.

The GPT-2 byte-to-Unicode mapping provides a printable Unicode representation for such byte values.

The conceptual process is:

```text
Raw Byte
    ↓
GPT-2 Mapping
    ↓
Printable Unicode Representation
```

**Important:**  
The mapping does not create additional byte values. There are still exactly 256 possible byte values. Unicode code points above 255 are only used as representations for some byte values.

---

## 8. Why Is `gpt2_bytes_to_unicode()` Useful?

Without a special representation, some byte sequences can be difficult to display.

For example, byte `0` corresponds to a control character. A direct conversion:

```python
chr(0)
```

does not produce a visually useful representation.

The GPT-2 mapping solves this representation problem.

Conceptually:

```text
Byte 0
  ↓
Printable Unicode representation
```

while preserving the original byte identity.

Therefore, the mapping should be understood as a byte representation layer, not as a separate tokenization algorithm.

---

## 9. Training Step 2 — Pre-tokenization

After the text is prepared, the next step is pre-tokenization.

Pre-tokenization divides the raw text into smaller pieces before BPE merging begins.

For example:

```text
Hello, world! I'm learning BPE.
```

can be divided approximately into:

```text
Hello
,
 world
!
 I
'm
 learning
 BPE
.
```

These pieces are called pre-tokens.

---

## 10. Why Is Pre-tokenization Needed?

Pre-tokenization establishes boundaries before BPE starts learning merges.

BPE operates within these boundaries.

For example:

```text
"hello" | " world"
```

contains two separate pre-tokens.

The tokenizer can perform BPE operations inside:

```text
hello
```

and:

```text
 world
```

independently.

The merge process does not simply combine arbitrary text from across the boundary.

Therefore:

- **Pre-tokenization** → Establish boundaries  
- **BPE** → Learn merges inside those boundaries  

---

## 11. GPT-2-Style Pre-tokenization

The implementation uses a GPT-2-style regular expression:

```python
GPT2_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)
```

This pattern handles different categories of text such as:

- Letters  
- Numbers  
- Punctuation  
- Whitespace  
- Common English contractions  

For example:

```text
I'm
```

can be separated into:

```text
I
'm
```

The implementation uses the Python regex package because the pattern uses Unicode properties such as:

- `\p{L}`  
- `\p{N}`  

---

## 12. Pre-tokenization vs BPE

Pre-tokenization and BPE are different operations.

**Pre-tokenization** determines:

- Where should the text be divided before BPE?

```text
Text
 ↓
Pre-tokenization
 ↓
Pre-tokens
```

**BPE** determines:

- Which adjacent units should be merged?

```text
Pre-token
 ↓
BPE merges
 ↓
BPE tokens
```

Therefore:

- **Pre-tokenization** = Establish boundaries  
- **BPE** = Learn and apply merges  

---

## 13. Training Step 3 — UTF-8 Conversion

After pre-tokenization, each pre-token is converted into UTF-8 bytes.

For example:

```text
cat
```

becomes:

```text
[97][99][116]
```

because:

- `c → 99`  
- `a → 97`  
- `t → 116`  

For Unicode characters, UTF-8 may require multiple bytes.

For example:

```text
🙃
```

becomes:

```text
[131][153][159][240]
```

Therefore:

```text
Unicode Text
      ↓
Pre-token
      ↓
UTF-8 Encoding
      ↓
Byte Sequence
```

---

## 14. Why UTF-8 Is Important

BPE in this implementation operates on bytes, not directly on Python Unicode characters.

The workflow is therefore:

```text
Unicode text
      ↓
UTF-8
      ↓
Bytes
      ↓
BPE
```

UTF-8 provides a standard byte representation for Unicode text.

For example:

```text
"A"
 ↓

```

while:

```text
"🙃"
 ↓
[131][153][159][240]
```

BPE can then operate on these resulting byte sequences.

---

## 15. Training Step 4 — Initial Vocabulary

BPE starts with all 256 possible byte values.

Conceptually:

```python
vocab = {
    0: b'\x00',
    1: b'\x01',
    ...
    65: b'A',
    ...
    255: b'\xff'
}
```

The initial state is:

- Vocabulary size = 256  
- Merge rules = 0  

Every byte initially represents one token.

For example:

```text
"cat"
```

becomes:

```text
c + a + t
```

or numerically:

```text
99 + 97 + 116
```

---

## 16. Training Step 5 — Count Adjacent Pairs

The tokenizer examines adjacent token pairs.

Suppose a pre-token contains:

```text
l o w
```

The adjacent pairs are:

- `(l, o)`  
- `(o, w)`  

The tokenizer counts how frequently each pair occurs throughout the training corpus.

For example:

```text
Pair          Frequency
-----------------------
(o, w)           120
(l, o)            95
(e, r)            80
```

The frequency represents how often the pair appears as adjacent tokens in the training data.

---

## 17. Training Step 6 — Select the Best Pair

The tokenizer selects the most frequent adjacent pair.

For example:

```text
(o, w)
```

would be selected if it has the highest frequency.

If multiple pairs have the same frequency, the required tie-breaking rule is applied.

In this implementation, ties are resolved using the lexicographically greater pair.

Therefore:

- Highest frequency  
- Tie? → Lexicographically greater pair  

The selected pair becomes the next BPE merge.

---

## 18. Training Step 7 — Merge the Pair

Suppose the selected pair is:

```text
(o, w)
```

The tokenizer combines the two tokens:

```text
o + w
 ↓
ow
```

At the byte level:

```text
b"o" + b"w"
       ↓
    b"ow"
```

A new vocabulary entry is created.

If the current vocabulary contains 256 tokens, the new token receives:

```text
ID = 256
```

Therefore:

```text
256 → b"ow"
```

The vocabulary size becomes:

```text
257
```

---

## 19. Training Step 8 — Update the Vocabulary

Every learned merge creates one new vocabulary entry.

For example:

```text
256 → b"ow"
257 → b"low"
258 → b" low"
```

The vocabulary therefore grows:

```text
256
 ↓
257
 ↓
258
 ↓
259
 ↓
...
```

The target vocabulary size determines when the training process stops.

---

## 20. Training Step 9 — Update the Merge Rules

The merge is also recorded in an ordered merge list.

For example:

```python
merges = [
    (b"o", b"w")
]
```

This means:

```text
o + w → ow
```

If another merge is learned:

```text
l + ow → low
```

the merge list becomes:

```python
merges = [
    (b"o", b"w"),
    (b"l", b"ow")
]
```

The order is important because it determines merge priority during encoding.

---

## 21. Training Step 10 — Update Pair Statistics

After a merge is performed, the token sequences have changed.

For example:

**Before:**

```text
l o w e r
```

**After:**

```text
l ow e r
```

The adjacent pairs are now different.

Therefore, pair statistics must be updated.

Conceptually:

```text
Old token sequence
       ↓
Merge selected pair
       ↓
New token sequence
       ↓
Update affected pair counts
```

The optimized implementation updates pair frequencies rather than recounting the entire corpus from scratch after every merge.

---

## 22. Training Step 11 — Repeat

The BPE training process repeats:

- Count pairs  
- Select best pair  
- Merge pair  
- Update vocabulary  
- Record merge rule  
- Update pair statistics  
- Repeat  

For example:

```text
l o w
```

First:

```text
o + w → ow
```

Result:

```text
l ow
```

Then:

```text
l + ow → low
```

Result:

```text
low
```

The tokenizer gradually learns larger and more frequent pieces.

---

## 23. When Does BPE Training Stop?

Training continues until the requested vocabulary size is reached.

For example:

```python
vocab_size = 10000
```

The tokenizer starts with:

- 256 byte tokens  

and adds learned vocabulary entries through BPE merges until the target size is reached, subject to special-token handling where applicable.

The final result is:

- Vocabulary  
- Ordered Merge Rules  

---

## 24. Training Output — Vocabulary

The vocabulary is represented as:

```python
vocab
```

Conceptually:

```text
Token ID → Token Bytes
```

Example:

```text
256 → b"ow"
257 → b"low"
258 → b" low"
```

The vocabulary answers:

> What byte sequence does this token ID represent?

---

## 25. Training Output — Merge Rules

The merge rules are represented as:

```python
merges
```

Example:

```python
merges = [
    (b"o", b"w"),
    (b"l", b"ow"),
    (b" ", b"low")
]
```

They describe how larger tokens were created.

For example:

```text
b"o" + b"w"
       ↓
    b"ow"
```

then:

```text
b"l" + b"ow"
       ↓
    b"low"
```

The merge list is ordered by the time each merge was learned.

---

## 26. Vocabulary vs Merge Rules

These two components have different responsibilities.

**Vocabulary** answers:

> What does this token ID represent?

Example:

```text
257 → b"low"
```

**Merge Rules** answer:

> How was this token created?

Example:

```text
b"l" + b"ow" → b"low"
```

Together:

- Vocabulary + Merge Rules  

define the trained tokenizer.

---

## 27. Example of the Complete Training Process

Consider the small corpus:

```text
low low lower
```

Initially:

```text
l o w
l o w
l o w e r
```

Suppose:

```text
o + w
```

is the most frequent pair.

The tokenizer learns:

```text
o + w → ow
```

The corpus representation becomes:

```text
l ow
l ow
l ow e r
```

Suppose the next best pair is:

```text
l + ow
```

The tokenizer learns:

```text
l + ow → low
```

Now the representation becomes:

```text
low
low
low e r
```

The tokenizer has learned reusable pieces from the corpus.

---

## 28. Phase 2 — Encoding

Once training is complete, the learned vocabulary and merge rules are used to encode new text.

Encoding converts:

```text
Text → Token IDs
```

The complete workflow is:

```text
Input Text
     ↓
Special Token Handling
     ↓
Pre-tokenization
     ↓
UTF-8 Conversion
     ↓
Initial Byte Tokens
     ↓
Apply Learned Merge Rules
     ↓
Final Token IDs
```

---

## 29. Encoding Step 1 — Special Token Handling

Special tokens are reserved tokens with specific meanings.

For example:

```text
<|endoftext|>
```

should be treated as one complete special token rather than being broken into ordinary BPE pieces.

For example:

```text
Hello <|endoftext|> world
```

is conceptually divided into:

```text
Hello
<|endoftext|>
 world
```

The special token receives its own token ID.

Special tokens are treated separately from normal BPE processing.

---

## 30. Encoding Step 2 — Pre-tokenization

Normal text is passed through the GPT-2-style pre-tokenizer.

For example:

```text
Hello world!
```

can become:

```text
Hello
 world
!
```

Each pre-token is then processed independently.

---

## 31. Encoding Step 3 — UTF-8 Conversion

Each pre-token is converted into UTF-8 bytes.

For example:

```text
low
```

becomes:

```text
[108][111][119]
```

These correspond to:

- `l → 108`  
- `o → 111`  
- `w → 119`  

---

## 32. Encoding Step 4 — Start with Byte Tokens

Initially, each byte is represented by its corresponding vocabulary token.

For:

```text
low
```

the initial representation is:

```text
l + o + w
```

or:

```text
108 + 111 + 119
```

The tokenizer then checks which learned merge rules can be applied.

---

## 33. Encoding Step 5 — Apply Learned Merge Rules

Suppose training learned:

```text
o + w → ow
l + ow → low
```

Starting with:

```text
l o w
```

the tokenizer first applies:

```text
o + w
 ↓
ow
```

Result:

```text
l ow
```

Then:

```text
l + ow
 ↓
low
```

Result:

```text
low
```

The resulting token is mapped to its vocabulary ID.

For example:

```text
low → 257
```

Therefore:

```text
"low"
   ↓

```

---

## 34. Encoding Merge Priority

The order in which merges were learned is important.

Suppose the merge list is:

1. `o + w → ow`  
2. `l + ow → low`  

The encoder uses these learned priorities when deciding which valid merge to apply.

Therefore, encoding is not simply:

> Merge any pair randomly  

Instead, it follows the learned BPE merge ordering.

Conceptually:

```text
Input bytes
     ↓
Find valid learned pairs
     ↓
Choose highest-priority merge
     ↓
Merge
     ↓
Repeat
```

---

## 35. Encoding Output

The final result is a list of integer token IDs.

Example:

```text
[257][258][891]
```

These IDs are what a language model consumes.

The overall pipeline is:

```text
Human-readable text
        ↓
Tokenizer
        ↓
Token IDs
        ↓
Language Model
```

---

## 36. `encode_iterable()`

The tokenizer also provides:

```python
encode_iterable()
```

This allows text from an iterable to be encoded incrementally.

Conceptually:

```text
Input iterable
      ↓
Read one piece
      ↓
Encode
      ↓
Yield token IDs
      ↓
Read next piece
      ↓
Continue
```

This is useful when working with large amounts of text because the entire output does not have to be materialized as one large list at once.

---

## 37. Phase 3 — Decoding

Decoding performs the reverse operation of encoding.

It converts:

```text
Token IDs → Text
```

The complete workflow is:

```text
Token IDs
    ↓
Vocabulary Lookup
    ↓
Token Bytes
    ↓
Concatenate Bytes
    ↓
UTF-8 Decoding
    ↓
Text
```

---

## 38. Decoding Step 1 — Vocabulary Lookup

Each token ID is looked up in the vocabulary.

For example:

```text
257 → b"low"
258 → b" world"
```

Given:

```text
[257][258]
```

the tokenizer retrieves:

```text
b"low"
b" world"
```

---

## 39. Decoding Step 2 — Convert Token IDs to Bytes

The vocabulary maps every token ID to a byte sequence.

Therefore:

```text
Token IDs
    ↓
Vocabulary
    ↓
Token Bytes
```

Example:

```text
257
 ↓
b"low"
```

and:

```text
258
 ↓
b" world"
```

---

## 40. Decoding Step 3 — Concatenate Token Bytes

The individual byte sequences are concatenated.

For example:

```text
b"low"
+
b" world"
```

becomes:

```text
b"low world"
```

The tokenizer does not decode each token independently.

Instead, the token bytes are combined first.

This is important for Unicode because a Unicode character may consist of multiple UTF-8 bytes.

---

## 41. Decoding Step 4 — UTF-8 Decoding

After all token bytes have been concatenated, the resulting byte sequence is decoded using UTF-8.

For example:

```text
[131][153][159][240]
```

becomes:

```text
🙃
```

The complete process is:

```text
Token IDs
    ↓
Token Bytes
    ↓
Concatenated Bytes
    ↓
UTF-8
    ↓
Unicode Text
```

---

## 42. Why Concatenate Before UTF-8 Decoding?

UTF-8 characters can consist of multiple bytes.

For example:

```text
🙃
```

is represented as:

```text
[131][153][159][240]
```

If different tokens contain parts of this byte sequence, decoding each token separately could produce invalid Unicode.

Therefore, the correct conceptual workflow is:

```text
Token IDs
    ↓
Retrieve bytes
    ↓
Concatenate all bytes
    ↓
Decode the complete byte sequence
    ↓
Text
```

---

## 43. Complete Encoding Example

Suppose the input is:

```text
low world
```

The encoding process is:

```text
"low world"
      ↓
Pre-tokenization
      ↓
"low"
" world"
      ↓
UTF-8
      ↓
Byte sequences
      ↓
Apply learned BPE merges
      ↓
Vocabulary lookup
      ↓
Token IDs
```

For example:

```text
"low world"
      ↓
[257][258]
```

where:

- `257 → b"low"`  
- `258 → b" world"`  

---

## 44. Complete Decoding Example

Starting with:

```text
[257][258]
```

the decoding process is:

```text
[257][258]
      ↓
Vocabulary lookup
      ↓
b"low"
b" world"
      ↓
Concatenate
      ↓
b"low world"
      ↓
UTF-8 decoding
      ↓
"low world"
```

Therefore:

```text
Token IDs
    ↓
Bytes
    ↓
Text
```

---

## 45. Complete BPE Workflow

The complete tokenizer can be represented as:

```text
                         TRAINING
                            │
                            ▼
                    Training Corpus
                            │
                            ▼
                    Pre-tokenization
                            │
                            ▼
                       UTF-8 Bytes
                            │
                            ▼
                   Initial 256 Tokens
                            │
                            ▼
                   Count Pair Frequencies
                            │
                            ▼
                    Select Best Pair
                            │
                            ▼
                       Merge Pair
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          Update Vocabulary     Record Merge Rule
                  │                   │
                  └─────────┬─────────┘
                            │
                            ▼
                         Repeat
                            │
                            ▼
                  Trained Tokenizer
                   ┌────────┴────────┐
                   ▼                 ▼
              Vocabulary          Merges
                   │                 │
                   └────────┬────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
             ENCODING                DECODING
                │                       ▲
                ▼                       │
              Text                   Token IDs
                │                       │
                ▼                       │
         Special Tokens                 │
                │                       │
                ▼                       │
         Pre-tokenization                │
                │                       │
                ▼                       │
             UTF-8 Bytes                 │
                │                       │
                ▼                       │
          Apply BPE Merges               │
                │                       │
                ▼                       │
           Token IDs ───────────────────┘
```

---

## 46. Training vs Encoding vs Decoding

| Phase     | Input           | Main Operations                                | Output                    |
|-----------|------------------|------------------------------------------------|---------------------------|
| Training  | Training corpus  | Pre-tokenization, UTF-8, pair counting, merging| Vocabulary + merge rules  |
| Encoding  | New text         | Pre-tokenization, UTF-8, learned merges        | Token IDs                 |
| Decoding  | Token IDs        | Vocabulary lookup, byte concatenation, UTF-8   | Text                      |

---

## 47. Important Components

- **Vocabulary**  
  Maps: `Token ID → Token Bytes`  
  Example: `257 → b"low"`  

- **Merge Rules**  
  Store: `Pair of tokens → Learned merge order`  
  Example: `b"l" + b"ow"`  

- **Pre-tokenization**  
  Splits raw text into pieces before BPE.  

- **UTF-8**  
  Converts Unicode text into bytes.  

- **BPE Training**  
  Learns frequently occurring combinations.  

- **Encoding**  
  Uses the trained tokenizer.  

- **Decoding**  
  Reconstructs text.  

---

## 48. Important Difference: Training vs Encoding

Training and encoding both perform BPE-related operations, but their purposes are different.

**Training** discovers the merge rules:

```text
Corpus
 ↓
Count pairs
 ↓
Select pair
 ↓
Merge
 ↓
Learn rule
```

**Encoding** uses the already learned merge rules:

```text
New text
 ↓
Initial byte tokens
 ↓
Use learned rules
 ↓
Token IDs
```

Therefore:

- **Training** = Learn the tokenizer  
- **Encoding** = Use the tokenizer  

---

## 49. Important Difference: Encoding vs Decoding

**Encoding:**

```text
Text → Token IDs
```

**Decoding:**

```text
Token IDs → Text
```

More precisely:

**ENCODING**

```text
Text
 ↓
Pre-tokenization
 ↓
UTF-8
 ↓
BPE merges
 ↓
Token IDs
```

**DECODING**

```text
Token IDs
 ↓
Vocabulary lookup
 ↓
Bytes
 ↓
Concatenate
 ↓
UTF-8
 ↓
Text
```

---

## 50. Round-Trip Property

A correctly implemented tokenizer should be able to encode text and then decode the resulting token IDs back to the original text for valid supported input.

Conceptually:

```text
Original Text
     ↓
   Encode
     ↓
 Token IDs
     ↓
   Decode
     ↓
Original Text
```

For example:

```python
text = "Hello world"

ids = tokenizer.encode(text)

decoded = tokenizer.decode(ids)
```

Conceptually:

```text
"Hello world"
      ↓
[...token IDs...]
      ↓
"Hello world"
```

This is commonly called a round trip.

---

## 51. Project Structure

```text
.
├── common.py
├── train_bpe.py
├── tokenizer.py
└── test_tokenizer.py
```

---

## 52. `common.py`

Contains common tokenizer functionality such as the GPT-2-style pre-tokenization logic.

Main functionality:

- `pretokenize(text)`  

The module also contains the GPT-2 byte-to-Unicode mapping:

- `gpt2_bytes_to_unicode()`  

---

## 53. `train_bpe.py`

Responsible for training the BPE tokenizer.

Main responsibility:

```text
Training Corpus
      ↓
BPE Training
      ↓
Vocabulary + Merge Rules
```

The main training function is conceptually:

```python
vocab, merges = train_bpe(text, vocab_size)
```

---

## 54. `tokenizer.py`

Contains the tokenizer implementation used after training.

Main operations include:

- `tokenizer.encode(text)`  
- `tokenizer.decode(ids)`  
- `tokenizer.encode_iterable(iterable)`  

The tokenizer uses the trained:

- Vocabulary  
- Merge Rules  

to convert between text and token IDs.

---

## 55. `test_tokenizer.py`

Contains tests for tokenizer behavior.

The tests cover functionality such as:

- Basic encoding  
- Decoding  
- BPE merges  
- Unicode handling  
- Special tokens  
- Iterable encoding  
- Round-trip behavior  

---

## 56. Example Usage

**Training:**

```python
from pathlib import Path
from train_bpe import train_bpe

text = Path("corpus.en").read_text(encoding="utf-8")

vocab, merges = train_bpe(
    text,
    vocab_size=300
)

print("Vocabulary size:", len(vocab))
print("Number of merges:", len(merges))
```

**Using the trained tokenizer:**

```python
from tokenizer import Tokenizer

tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=["<|endoftext|>"]
)

text = "Hello world!"

ids = tokenizer.encode(text)

print(ids)

decoded = tokenizer.decode(ids)

print(decoded)
```

---

## 57. Testing

Run the tokenizer tests with:

```bash
pytest -q test_tokenizer.py
```

The current standalone tokenizer test suite contains:

- 23 tests  

All tests currently pass:

- 23 passed  

---

## 58. Key Concepts Summary

| Concept                | Purpose                                      |
|------------------------|----------------------------------------------|
| Byte                   | Fundamental initial BPE unit                 |
| UTF-8                  | Converts Unicode text into bytes             |
| `gpt2_bytes_to_unicode()` | Provides printable Unicode representations for byte values |
| Pre-tokenization       | Establishes boundaries before BPE            |
| Vocabulary             | Maps token IDs to byte sequences             |
| Merge rule             | Describes a learned token combination        |
| Pair frequency         | Determines which pair is most useful to merge|
| BPE training           | Learns vocabulary and merge rules            |
| Encoding               | Converts text into token IDs                 |
| Decoding               | Converts token IDs back into text            |
| Special token          | Reserved token handled separately            |
| `encode_iterable()`    | Supports incremental encoding                |

---

## 59. Final Mental Model

The easiest way to understand the entire tokenizer is:

```text
                    BPE TOKENIZER
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
       TRAINING        ENCODING        DECODING
          │               │               │
          │               │               │
      Learn rules      Use rules       Reverse lookup
          │               │               │
          ▼               ▼               ▼
       Corpus            Text          Token IDs
          │               │               │
          ▼               ▼               ▼
   Pre-tokenization  Pre-tokenization  Vocabulary
          │               │               │
          ▼               ▼               ▼
       UTF-8           UTF-8            Bytes
          │               │               │
          ▼               ▼               ▼
       Bytes          BPE merges      Concatenate
          │               │               │
          ▼               ▼               ▼
    Pair counting      Token IDs        UTF-8
          │                               │
          ▼                               ▼
     Merge pairs                         Text
          │
          ▼
   Vocabulary +
   Merge Rules
```

The complete idea can be reduced to:

**TRAINING**

```text
Corpus
  ↓
Pre-tokenize
  ↓
UTF-8
  ↓
256 byte vocabulary
  ↓
Count frequent pairs
  ↓
Merge best pair
  ↓
Update vocabulary
  ↓
Record merge rule
  ↓
Repeat
```

**ENCODING**

```text
Text
  ↓
Pre-tokenize
  ↓
UTF-8
  ↓
Byte tokens
  ↓
Apply learned merge rules
  ↓
Token IDs
```

**DECODING**

```text
Token IDs
  ↓
Vocabulary lookup
  ↓
Token bytes
  ↓
Concatenate bytes
  ↓
UTF-8 decode
  ↓
Text
```

---

## 60. Final Takeaway

A BPE tokenizer is essentially a learned system for compressing text into reusable subword tokens.

It starts from:

- 256 byte tokens  

and learns larger units by repeatedly merging frequent adjacent pairs.

The three phases are:

1. **Training**  
   Learn the vocabulary and merge rules.  

2. **Encoding**  
   Convert text into token IDs using the learned rules.  

3. **Decoding**  
   Convert token IDs back into the original text.  

The most important relationship is:

```text
TRAINING
    ↓
Vocabulary + Ordered Merge Rules
    ↓
ENCODING
    ↓
Token IDs
    ↓
DECODING
    ↓
Text
```

In short:

- **BPE TRAINING** = Learn how text should be tokenized  
- **ENCODING** = Apply what was learned  
- **DECODING** = Reconstruct the text  