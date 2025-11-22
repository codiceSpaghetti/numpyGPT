import re


class WordTokenizer:
    def __init__(self, min_freq: int = 1, max_vocab_size: int | None = None) -> None:
        self.min_freq: int = min_freq
        self.max_vocab_size: int | None = max_vocab_size
        self.words: list[str] = []
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def build_vocab(self, text: str) -> None:
        text = text.replace('\n', ' <|newline|> ')
        text = text.replace('\t', ' <|tab|> ')
        text = text.replace('\r', ' <|carriage_return|> ')

        # Tokenize text into:
        # 1. Special tokens with format <|...|>   -> <\|[^|]+\|>
        # 2. Words (alphanumerics/underscores)    -> \b\w+\b
        # 3. Punctuation/non-word characters      -> [^\s\w]
        tokens = re.findall(r"<\|[^|]+\|>|\b\w+\b|[^\s\w]", text)
        tokens = [token.lower() if not (token.startswith('<|') and token.endswith('|>')) else token for token in tokens]

        word_counts = {}
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1

        filtered_words = [word for word, count in word_counts.items() if count >= self.min_freq]
        sorted_words = sorted(filtered_words, key=lambda x: (-word_counts[x], x))  # sort by frequency, then alphabetically

        if self.max_vocab_size:
            sorted_words = sorted_words[:self.max_vocab_size - 4]

        self.words = ['<pad>', '<unk>', '<bos>', '<eos>'] + sorted_words
        self.stoi = {word: i for i, word in enumerate(self.words)}
        self.itos = {i: word for i, word in enumerate(self.words)}

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        text = text.replace('\n', ' <|newline|> ')
        text = text.replace('\t', ' <|tab|> ')
        text = text.replace('\r', ' <|carriage_return|> ')

        tokens = re.findall(r"<\|[^|]+\|>|\b\w+\b|[^\s\w]", text)
        tokens = [token.lower() if not (token.startswith('<|') and token.endswith('|>')) else token for token in tokens]

        indices = [self.stoi.get(token, 1) for token in tokens]  # 1 is the index of <unk>
        if add_bos:
            indices = [2] + indices
        if add_eos:
            indices = indices + [3]
        return indices

    def decode(self, tokens: list[int]) -> str:
        words = [self.itos[t] for t in tokens if t in self.itos]

        for special in ['<bos>', '<eos>', '<pad>', '<unk>']:
            while special in words:
                words.remove(special)

        result = []
        for i, word in enumerate(words):
            if word == '<|newline|>':
                result.append('\n')
            elif word == '<|tab|>':
                result.append('\t')
            elif word == '<|carriage_return|>':
                result.append('\r')
            else:
                result.append(word)
                if i < len(words) - 1:
                    next_word = words[i + 1]
                    if not next_word.startswith('<|') and next_word not in '\'.,;!?)"':
                        result.append(' ')

        text = ''.join(result)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r" ([\'.,;:!?)])", r'\1', text)
        text = re.sub(r"\b([A-Za-z]+) ?' ?(ll|re|ve|d|s|t|m)\b", r"\1'\2", text)
        text = re.sub(r"([a-z]) :", r'\1:', text)
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r' *\t *', '\t', text)
        text = re.sub(r' *\r *', '\r', text)
        return text.strip()

    @property
    def vocab_size(self) -> int:
        return len(self.words)

    @property
    def eos_token_id(self) -> int:
        return 3

    @property
    def bos_token_id(self) -> int:
        return 2
