import pytest

from tokenizer import Tokenizer


# ============================================================
# Small test vocabulary
# ============================================================

def make_test_tokenizer():
    # Initial byte vocabulary
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    # Learned BPE tokens
    vocab[256] = b"he"
    vocab[257] = b"ll"
    vocab[258] = b"hell"

    # Special token
    vocab[259] = b"<|endoftext|>"

    # Learned merges, in training order
    merges = [
        (b"h", b"e"),       # -> b"he"
        (b"l", b"l"),       # -> b"ll"
        (b"he", b"ll"),     # -> b"hell"
    ]

    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"]
    )


# ============================================================
# Basic tests
# ============================================================

def test_empty():

    tokenizer = make_test_tokenizer()

    text = ""

    ids = tokenizer.encode(text)

    assert ids == []

    assert tokenizer.decode(ids) == text


def test_single_character():

    tokenizer = make_test_tokenizer()

    text = "a"

    ids = tokenizer.encode(text)

    assert ids == [97]

    assert tokenizer.decode(ids) == text


def test_bpe_merge():

    tokenizer = make_test_tokenizer()

    text = "hello"

    ids = tokenizer.encode(text)

    # "hell" -> token 258
    # "o"    -> byte token 111
    assert ids == [258, 111]

    assert tokenizer.decode(ids) == text


def test_normal_text_roundtrip():

    tokenizer = make_test_tokenizer()

    text = "hello world"

    ids = tokenizer.encode(text)

    decoded = tokenizer.decode(ids)

    assert decoded == text


# ============================================================
# Unicode
# ============================================================

def test_unicode():

    tokenizer = make_test_tokenizer()

    text = "Hello 🙃"

    ids = tokenizer.encode(text)

    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_unicode_multiple():

    tokenizer = make_test_tokenizer()

    text = "Héllò hôw are ü? 🙃"

    ids = tokenizer.encode(text)

    decoded = tokenizer.decode(ids)

    assert decoded == text


# ============================================================
# Special tokens
# ============================================================

def test_special_token():

    tokenizer = make_test_tokenizer()

    text = "hello <|endoftext|> world"

    ids = tokenizer.encode(text)

    # Special token must be ONE token
    assert 259 in ids

    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_multiple_special_tokens():

    tokenizer = make_test_tokenizer()

    text = (
        "hello "
        "<|endoftext|>"
        "<|endoftext|>"
        " world"
    )

    ids = tokenizer.encode(text)

    assert ids.count(259) == 2

    assert tokenizer.decode(ids) == text


def test_special_token_at_beginning():

    tokenizer = make_test_tokenizer()

    text = "<|endoftext|>hello"

    ids = tokenizer.encode(text)

    assert ids[0] == 259

    assert tokenizer.decode(ids) == text


def test_special_token_at_end():

    tokenizer = make_test_tokenizer()

    text = "hello<|endoftext|>"

    ids = tokenizer.encode(text)

    assert ids[-1] == 259

    assert tokenizer.decode(ids) == text


# ============================================================
# Overlapping special tokens
# ============================================================

def test_overlapping_special_tokens():

    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    vocab[256] = b"<|endoftext|>"
    vocab[257] = b"<|endoftext|><|endoftext|>"

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=[],
        special_tokens=[
            "<|endoftext|>",
            "<|endoftext|><|endoftext|>",
        ]
    )

    text = (
        "hello "
        "<|endoftext|><|endoftext|>"
        " world"
    )

    ids = tokenizer.encode(text)

    # Longest special token should be selected
    assert 257 in ids

    # It should NOT become two 256 tokens
    assert ids.count(257) == 1

    assert tokenizer.decode(ids) == text


# ============================================================
# Iterable encoding
# ============================================================

def test_encode_iterable():

    tokenizer = make_test_tokenizer()

    texts = [
        "hello",
        "world",
        "this is BPE",
    ]

    ids = list(
        tokenizer.encode_iterable(texts)
    )

    decoded = tokenizer.decode(ids)

    # encode_iterable receives each string separately,
    # so joining them reproduces the concatenated input.
    assert decoded == "".join(texts)


def test_encode_iterable_with_special_tokens():

    tokenizer = make_test_tokenizer()

    texts = [
        "hello ",
        "<|endoftext|>",
        " world",
    ]

    ids = list(
        tokenizer.encode_iterable(texts)
    )

    assert 259 in ids

    decoded = tokenizer.decode(ids)

    assert decoded == "".join(texts)


# ============================================================
# Decode invalid UTF-8
# ============================================================

def test_decode_invalid_utf8():

    tokenizer = make_test_tokenizer()

    # 0xff is not valid UTF-8 by itself
    text = tokenizer.decode([255])

    assert text == "�"


# ============================================================
# Round-trip test
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "hello",
        "hello world",
        "construction and repair of highways and roads",
        "Héllò hôw are ü? 🙃",
        "hello <|endoftext|> world",
        "<|endoftext|>hello",
        "hello<|endoftext|>",
    ]
)
def test_roundtrip(text):

    tokenizer = make_test_tokenizer()

    ids = tokenizer.encode(text)

    decoded = tokenizer.decode(ids)

    assert decoded == text