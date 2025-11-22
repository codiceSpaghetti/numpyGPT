from typing import Protocol


class Tokenizer(Protocol):
    """Protocol defining the interface for all tokenizers."""

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...

    @property
    def eos_token_id(self) -> int:
        """Return the end-of-sequence token ID."""
        ...

    @property
    def bos_token_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        ...

    def build_vocab(self, text: str | list[str]) -> None:
        """Build vocabulary from text."""
        ...

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...
