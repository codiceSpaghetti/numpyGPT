from collections import Counter


class CharTokenizer:
    def __init__(self, special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']):
        self.special_tokens = special_tokens
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def tokenize(self, text):
        return list(text)

    def build_vocab(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        all_chars = []
        for text in texts:
            all_chars.extend(self.tokenize(text))

        char_counts = Counter(all_chars)
        unique_chars = sorted(char_counts.keys())

        idx = 0
        self.char_to_idx = {}
        self.idx_to_char = {}

        for token in self.special_tokens:
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
            idx += 1

        for char in unique_chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1

        self.vocab_size = len(self.char_to_idx)

    def encode(self, text):
        chars = self.tokenize(text)
        indices = [self.char_to_idx.get(char, self.char_to_idx['<unk>']) for char in chars]
        return [self.char_to_idx['<bos>']] + indices

    def decode(self, indices):
        chars = [self.idx_to_char.get(idx, '<unk>') for idx in indices]
        if chars and chars[0] == '<bos>':
            chars = chars[1:]
        if chars and chars[-1] == '<eos>':
            chars = chars[:-1]
        return ''.join(chars)

    @property
    def eos_token_id(self):
        return self.char_to_idx['<eos>']
