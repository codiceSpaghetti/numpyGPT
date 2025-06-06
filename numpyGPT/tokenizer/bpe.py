import re
from collections import Counter, defaultdict

"""
BPE learns subword vocab by iteratively merging most frequent adjacent pairs
Start: "hello" -> ['h','e','l','l','o'] 
Find 'l','l' is most frequent pair -> merge to 'll'
Result: "hello" -> ['h','e','ll','o']
Repeat until vocab_size reached => meaningful subwords emerge naturally.
"""


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.stoi = {}
        self.itos = {}
        self.merges = []
        self.merge_ranks = {}

    def _get_word_freqs(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        word_freqs = Counter()
        for text in texts:
            text = text.replace('\n', ' <|newline|> ')
            text = text.replace('\t', ' <|tab|> ')
            text = text.replace('\r', ' <|carriage_return|> ')

            # Tokenize text into:
            # 1. Special tokens with format <|...|>   -> <\|[^|]+\|>
            # 2. Words (alphanumerics/underscores)    -> \b\w+\b
            # 3. Punctuation/non-word characters      -> [^\s\w]
            tokens = re.findall(r"<\|[^|]+\|>|\b\w+\b|[^\s\w]", text)
            tokens = [token if not (token.startswith('<|') and token.endswith('|>')) else token for token in tokens]

            for token in tokens:
                word_freqs[token] += 1
        return word_freqs

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _merge_word(self, word_tokens, pair):
        new_tokens = []
        i = 0
        while i < len(word_tokens):
            # If we find the pair, merge it
            if i < len(word_tokens) - 1 and word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2  # Skip both tokens
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        return new_tokens

    def build_vocab(self, texts):
        word_freqs = self._get_word_freqs(texts)

        # Step 1: Initialize with character-level tokens
        vocab = {}
        base_chars = set()

        for word, freq in word_freqs.items():
            if word.startswith('<|') and word.endswith('|>'):
                # Special tokens stay as single units
                vocab[word + ' </w>'] = freq
                base_chars.add(word)
            else:
                # Split into characters + end-of-word marker
                char_list = list(word) + ['</w>']
                vocab[' '.join(char_list)] = freq
                base_chars.update(word)

        base_chars.add('</w>')

        # Step 2: Calculate how many merges we can do
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        base_vocab_size = len(special_tokens) + len(base_chars)
        available_for_merges = max(0, self.vocab_size - base_vocab_size)

        # Step 3: Iteratively find and apply merges
        for i in range(available_for_merges):
            pairs = defaultdict(int)
            # Count all adjacent pairs across all words
            for word, freq in vocab.items():
                word_pairs = self._get_pairs(word.split())
                for pair in word_pairs:
                    if not any(p.startswith('<|') and '|>' in p for p in pair):
                        pairs[pair] += freq

            if not pairs:
                break

            # Find most frequent pair, that will be merged in this iteration
            best_pair = max(pairs, key=pairs.get)

            # Update vocab with the merged pair
            new_vocab = {}
            for word, freq in vocab.items():
                word_tokens = word.split()
                merged_tokens = self._merge_word(word_tokens, best_pair)
                new_vocab[' '.join(merged_tokens)] = freq

            vocab = new_vocab

            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = i  # Remember when this was learned for encoding and decoding

         # Step 4: Add special tokens, base chars to the vocab and pairs to the vocab
        all_tokens = special_tokens + sorted(list(base_chars))
        for pair in self.merges:
            all_tokens.append(''.join(pair))

        all_tokens = all_tokens[:self.vocab_size]

        self.stoi = {token: idx for idx, token in enumerate(all_tokens)}
        self.itos = {idx: token for idx, token in enumerate(all_tokens)}
        self.vocab_size = len(self.stoi)

    def _tokenize_word(self, word):
        if word.startswith('<|') and word.endswith('|>'):
            return [word, '</w>']

        word_tokens = list(word) + ['</w>']

        while len(word_tokens) > 1:
            pairs = self._get_pairs(word_tokens)
            if not pairs:
                break

            best_pair = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float('inf')))
            if best_pair not in self.merge_ranks:
                break

            word_tokens = self._merge_word(word_tokens, best_pair)

        return word_tokens

    def encode(self, text, add_bos=False, add_eos=False):
        text = text.replace('\n', ' <|newline|> ')
        text = text.replace('\t', ' <|tab|> ')
        text = text.replace('\r', ' <|carriage_return|> ')

        tokens = re.findall(r"<\|[^|]+\|>|\b\w+\b|[^\s\w]", text)
        tokens = [token if not (token.startswith('<|') and token.endswith('|>')) else token for token in tokens]

        indices = []
        if add_bos:
            indices.append(2)  # <bos> index

        for token in tokens:
            word_tokens = self._tokenize_word(token)
            for word_token in word_tokens:
                indices.append(self.stoi.get(word_token, 1))  # 1 is <unk>

        if add_eos:
            indices.append(3)  # <eos> index

        return indices

    def decode(self, indices):
        tokens = [self.itos.get(idx, '<unk>') for idx in indices]

        # Remove special tokens
        for special in ['<bos>', '<eos>', '<pad>', '<unk>']:
            while special in tokens:
                tokens.remove(special)

        result = []
        current_word = []

        for token in tokens:
            if token == '<|newline|>':
                if current_word:
                    result.append(''.join(current_word))
                    current_word = []
                result.append('\n')
            elif token == '<|tab|>':
                if current_word:
                    result.append(''.join(current_word))
                    current_word = []
                result.append('\t')
            elif token == '<|carriage_return|>':
                if current_word:
                    result.append(''.join(current_word))
                    current_word = []
                result.append('\r')
            elif token == '</w>':
                if current_word:
                    result.append(''.join(current_word))
                    current_word = []
                result.append(' ')
            else:
                current_word.append(token)

        if current_word:
            result.append(''.join(current_word))

        text = ''.join(result)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r' *\t *', '\t', text)
        text = re.sub(r' *\r *', '\r', text)
        return text.strip()

    @property
    def eos_token_id(self):
        return 3

    @property
    def bos_token_id(self):
        return 2
