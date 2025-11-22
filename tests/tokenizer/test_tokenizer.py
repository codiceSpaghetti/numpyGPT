import unittest

from numpyGPT.tokenizer.bpe import BPETokenizer
from numpyGPT.tokenizer.char_level import CharTokenizer
from numpyGPT.tokenizer.word_level import WordTokenizer


class TestCharTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = CharTokenizer()

    def test_tokenize(self):
        text = "hello world"
        tokens = self.tokenizer.tokenize(text)
        expected = ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
        self.assertEqual(tokens, expected)

    def test_tokenize_with_newlines(self):
        text = "hello\nworld"
        tokens = self.tokenizer.tokenize(text)
        expected = ['h', 'e', 'l', 'l', 'o', '\n', 'w', 'o', 'r', 'l', 'd']
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

    def test_encode_decode_with_newlines(self):
        text = "hello\nworld\ntest"
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


class TestWordTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_tokenize(self):
        text = "Hello, world! How are you?"
        tokens = self.tokenizer.tokenize(text)
        expected = ['hello', ',', 'world', '!', 'how', 'are', 'you', '?']
        self.assertEqual(tokens, expected)

    def test_tokenize_with_newlines(self):
        text = "hello\nworld"
        tokens = self.tokenizer.tokenize(text)
        expected = ['hello', '<|newline|>', 'world']
        self.assertEqual(tokens, expected)

    def test_tokenize_with_tabs(self):
        text = "hello\tworld"
        tokens = self.tokenizer.tokenize(text)
        expected = ['hello', '<|tab|>', 'world']
        self.assertEqual(tokens, expected)

    def test_build_vocab_single_text(self):
        text = "hello world hello"
        self.tokenizer.build_vocab(text)

        expected_words = set(['hello', 'world'])
        actual_words = set([word for word in self.tokenizer.word_to_idx.keys()
                           if word not in self.tokenizer.special_tokens])
        self.assertEqual(actual_words, expected_words)

    def test_build_vocab_multiple_texts(self):
        texts = ["hello world", "world peace"]
        self.tokenizer.build_vocab(texts)

        expected_words = set(['hello', 'world', 'peace'])
        actual_words = set([word for word in self.tokenizer.word_to_idx.keys()
                           if word not in self.tokenizer.special_tokens])
        self.assertEqual(actual_words, expected_words)

    def test_build_vocab_with_newlines(self):
        text = "hello\nworld"
        self.tokenizer.build_vocab(text)

        self.assertIn('<|newline|>', self.tokenizer.word_to_idx)

    def test_encode_decode(self):
        text = "hello world"
        self.tokenizer.build_vocab(text)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)

    def test_encode_decode_with_newlines(self):
        text = "hello\nworld\ntest"
        self.tokenizer.build_vocab(text)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)

    def test_unknown_word(self):
        self.tokenizer.build_vocab("hello world")

        encoded = self.tokenizer.encode("unknown words")
        unk_idx = self.tokenizer.word_to_idx['<unk>']
        expected = [unk_idx, unk_idx]

        self.assertEqual(encoded, expected)

    def test_min_freq_filtering(self):
        tokenizer = WordTokenizer(min_freq=2)
        text = "hello hello world world test"
        tokenizer.build_vocab(text)

        expected_words = set(['hello', 'world'])
        actual_words = set([word for word in tokenizer.word_to_idx.keys()
                           if word not in tokenizer.special_tokens])
        self.assertEqual(actual_words, expected_words)

    def test_max_vocab_size(self):
        tokenizer = WordTokenizer(max_vocab_size=4)
        text = "hello world test example"
        tokenizer.build_vocab(text)

        self.assertEqual(tokenizer.vocab_size, 4)

    def test_vocab_size(self):
        text = "hello world"
        self.tokenizer.build_vocab(text)

        expected_size = len(set(self.tokenizer.tokenize(text))) + len(self.tokenizer.special_tokens)
        self.assertEqual(self.tokenizer.vocab_size, expected_size)

    def test_punctuation_handling(self):
        text = "Hello, world!"
        self.tokenizer.build_vocab(text)

        tokens = self.tokenizer.tokenize(text)
        self.assertIn(',', tokens)
        self.assertIn('!', tokens)
        self.assertIn('hello', tokens)
        self.assertIn('world', tokens)


class TestBPETokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = BPETokenizer(vocab_size=50)

    def test_get_word_freqs_single_text(self):
        text = "hello world hello"
        word_freqs = self.tokenizer._get_word_freqs(text)
        expected = {'hello': 2, 'world': 1}
        self.assertEqual(word_freqs, expected)

    def test_get_word_freqs_multiple_texts(self):
        texts = ["hello world", "hello test"]
        word_freqs = self.tokenizer._get_word_freqs(texts)
        expected = {'hello': 2, 'world': 1, 'test': 1}
        self.assertEqual(word_freqs, expected)

    def test_get_word_freqs_with_newlines(self):
        text = "hello\nworld"
        word_freqs = self.tokenizer._get_word_freqs(text)
        expected = {'hello': 1, '<|newline|>': 1, 'world': 1}
        self.assertEqual(word_freqs, expected)

    def test_get_pairs(self):
        word = ['h', 'e', 'l', 'l', 'o']
        pairs = self.tokenizer._get_pairs(word)
        expected = {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')}
        self.assertEqual(pairs, expected)

    def test_build_vocab_learns_merges(self):
        text = "hello hello hello world world"
        self.tokenizer.build_vocab(text)

        self.assertGreater(len(self.tokenizer.merges), 0)
        self.assertIsInstance(self.tokenizer.merges[0], tuple)
        self.assertEqual(len(self.tokenizer.merges[0]), 2)

    def test_build_vocab_creates_mappings(self):
        text = "hello world"
        self.tokenizer.build_vocab(text)

        self.assertGreater(len(self.tokenizer.token_to_idx), 0)
        self.assertEqual(len(self.tokenizer.token_to_idx), len(self.tokenizer.idx_to_token))
        self.assertEqual(self.tokenizer.vocab_size, len(self.tokenizer.token_to_idx))

    def test_build_vocab_with_newlines(self):
        text = "hello\nworld"
        self.tokenizer.build_vocab(text)

        self.assertIn('<|newline|>', self.tokenizer.token_to_idx)

    def test_encode_decode_simple(self):
        text = "hello world"
        self.tokenizer.build_vocab(text)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)

    def test_encode_decode_repeated_words(self):
        text = "hello hello world"
        self.tokenizer.build_vocab(text)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)

    def test_encode_decode_with_newlines(self):
        text = "hello\nworld\ntest"
        self.tokenizer.build_vocab(text)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)

    def test_unknown_tokens(self):
        self.tokenizer.build_vocab("hello world")

        encoded = self.tokenizer.encode("xyz unknown")
        unk_idx = self.tokenizer.token_to_idx['<unk>']

        self.assertIn(unk_idx, encoded)

    def test_special_tokens_included(self):
        text = "hello world"
        self.tokenizer.build_vocab(text)

        self.assertIn('<pad>', self.tokenizer.token_to_idx)
        self.assertIn('<unk>', self.tokenizer.token_to_idx)

    def test_vocab_size_constraint(self):
        tokenizer = BPETokenizer(vocab_size=30)
        text = "hello world test example quick brown fox jumps over lazy dog"
        tokenizer.build_vocab(text)

        self.assertLessEqual(tokenizer.vocab_size, 30)

    def test_merges_applied_in_order(self):
        text = "aaaa bbbb aaaa bbbb"
        self.tokenizer.build_vocab(text)

        word = "aaaa"
        tokenized = self.tokenizer._tokenize_word(' '.join(list(word)))

        self.assertIsInstance(tokenized, list)
        self.assertGreater(len(tokenized), 0)

    def test_word_boundary_preservation(self):
        text = "hello world test"
        self.tokenizer.build_vocab(text)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)
        self.assertIn(' ', decoded)

    def test_empty_text(self):
        self.tokenizer.build_vocab("hello world")

        encoded = self.tokenizer.encode("")
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(encoded, [])
        self.assertEqual(decoded, "")

    def test_single_character_words(self):
        text = "a b c d"
        self.tokenizer.build_vocab(text)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)


if __name__ == '__main__':
    unittest.main()


