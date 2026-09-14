import regex

GPT2_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


def pretokenize(chunk):
    return regex.findall(GPT2_REGEX, chunk)
