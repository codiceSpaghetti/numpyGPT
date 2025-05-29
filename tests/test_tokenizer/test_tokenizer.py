from numpyGPT.tokenizer.char_level import CharTokenizer
import unittest


class TestCharTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = CharTokenizer()

    def test_tokenize(self):
        text = "hello world"
        tokens = self.tokenizer.tokenize(text)
        expected = ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
        self.assertEqual(tokens, expected)

    def test_build_vocab_single_text(self):
        text = "hello"
        self.tokenizer.build_vocab(text)

        expected_chars = set(['h', 'e', 'l', 'o'])
        actual_chars = set([char for char in self.tokenizer.char_to_idx.keys()
                           if char not in self.tokenizer.special_tokens])
        self.assertEqual(actual_chars, expected_chars)

    def test_build_vocab_multiple_texts(self):
        texts = ["hello", "world"]
        self.tokenizer.build_vocab(texts)

        expected_chars = set(['h', 'e', 'l', 'o', 'w', 'r', 'd'])
        actual_chars = set([char for char in self.tokenizer.char_to_idx.keys()
                           if char not in self.tokenizer.special_tokens])
        self.assertEqual(actual_chars, expected_chars)

    def test_encode_decode(self):
        text = "hello world"
        self.tokenizer.build_vocab(text)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)

    def test_unknown_char(self):
        self.tokenizer.build_vocab("hello")

        encoded = self.tokenizer.encode("xyz")
        unk_idx = self.tokenizer.char_to_idx['<unk>']
        expected = [unk_idx, unk_idx, unk_idx]

        self.assertEqual(encoded, expected)

    def test_vocab_size(self):
        text = "hello"
        self.tokenizer.build_vocab(text)

        expected_size = len(set(text)) + len(self.tokenizer.special_tokens)
        self.assertEqual(self.tokenizer.vocab_size, expected_size)


if __name__ == '__main__':
    unittest.main()
