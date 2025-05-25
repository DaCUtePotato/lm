import json
import struct
from collections import Counter

MIN_FREQ = 10000

def build_vocab(text, min_freq):
    freq = Counter(text)
    vocab = {ch: i for i, (ch, f) in enumerate(freq.items()) if f >= min_freq}
    return vocab

def tokenize(text, vocab):
    return [vocab[ch] for ch in text if ch in vocab]

def main():
    with open("../dataset.txt", "r", encoding="utf-8") as f:
        data = f.read()

    vocab = build_vocab(data, MIN_FREQ)
    with open("../vocab.txt", "w", encoding="utf-8") as f:
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\n")

    token_ids = tokenize(data, vocab)
    with open("../tokens.bin", "wb") as f:
        for token in token_ids:
            f.write(struct.pack("I", token))  # 4-byte unsigned int

if __name__ == "__main__":
    main()

