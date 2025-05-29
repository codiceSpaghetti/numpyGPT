from numpyGPT.tokenizer.char_level import CharTokenizer
from numpyGPT.tokenizer.word_level import WordTokenizer
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


class TestWordTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_tokenize(self):
        text = "Hello, world! How are you?"
        tokens = self.tokenizer.tokenize(text)
        expected = ['hello', ',', 'world', '!', 'how', 'are', 'you', '?']
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

    def test_encode_decode(self):
        text = "hello world"
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


if __name__ == '__main__':
    unittest.main()
