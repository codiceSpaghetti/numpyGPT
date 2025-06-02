import re
from collections import Counter


class WordTokenizer:
    def __init__(self, min_freq=1, max_vocab_size=None, special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

    def tokenize(self, text):
        tokens = re.findall(r"\b\w+\b|[^\s\w]", text)
        tokens = [token.lower() for token in tokens]
        return tokens

    def build_vocab(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))

        token_counts = Counter(all_tokens)
        sorted_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))

        if self.max_vocab_size is not None:
            sorted_tokens = sorted_tokens[:self.max_vocab_size - len(self.special_tokens)]

        sorted_tokens = [token for token, count in sorted_tokens if count >= self.min_freq]

        idx = 0
        self.word_to_idx = {}
        self.idx_to_word = {}

        for token in self.special_tokens:
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            idx += 1

        for token in sorted_tokens:
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            idx += 1

        self.vocab_size = len(self.word_to_idx)

    def encode(self, text):
        tokens = self.tokenize(text)
        indices = [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]
        return [self.word_to_idx['<bos>']] + indices

    def decode(self, indices):
        tokens = [self.idx_to_word.get(idx, '<unk>') for idx in indices]
        if tokens and tokens[0] == '<bos>':
            tokens = tokens[1:]
        if tokens and tokens[-1] == '<eos>':
            tokens = tokens[:-1]
        return ' '.join(tokens)
