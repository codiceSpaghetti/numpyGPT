class CharTokenizer:
    def __init__(self) -> None:
        self.chars: list[str] = []
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def build_vocab(self, text: str) -> None:
        chars = sorted(set(text))
        self.chars = ['<pad>', '<unk>', '<bos>', '<eos>'] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        tokens = [self.stoi.get(ch, 1) for ch in text]
        if add_bos:
            tokens = [self.stoi['<bos>']] + tokens
        if add_eos:
            tokens = tokens + [self.stoi['<eos>']]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        chars = [self.itos[t] for t in tokens if t in self.itos]
        text = ''.join(chars)
        for special in ['<bos>', '<eos>', '<pad>', '<unk>']:
            text = text.replace(special, '')
        return text

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def eos_token_id(self) -> int:
        return 3

    @property
    def bos_token_id(self) -> int:
        return 2
