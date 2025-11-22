class CharTokenizer:
    def __init__(self) -> None:
        self.chars: list[str] = []
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def build_vocab(self, text: str | list[str]) -> None:
        if isinstance(text, list):
            text = "".join(text)

        # Use tokenize() to get tokens
        tokens = self.tokenize(text)
        chars = sorted(set(tokens))

        self.chars = ["<pad>", "<unk>", "<bos>", "<eos>"] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def char_to_idx(self) -> dict[str, int]:
        """Alias for stoi to match test expectations."""
        return self.stoi

    @property
    def special_tokens(self) -> list[str]:
        """Return list of special tokens."""
        return ["<pad>", "<unk>", "<bos>", "<eos>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token IDs by first tokenizing, then mapping to indices."""
        tokens = self.tokenize(text)
        indices = [self.stoi.get(ch, 1) for ch in tokens]  # 1 is <unk>

        if add_bos:
            indices = [self.stoi["<bos>"]] + indices
        if add_eos:
            indices = indices + [self.stoi["<eos>"]]
        return indices

    def decode(self, tokens: list[int]) -> str:
        chars = [self.itos[t] for t in tokens if t in self.itos]
        text = "".join(chars)
        for special in ["<bos>", "<eos>", "<pad>", "<unk>"]:
            text = text.replace(special, "")
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
