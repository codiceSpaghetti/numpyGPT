import re
from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.merges = []

    def _get_word_freqs(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        word_freqs = Counter()
        for text in texts:
            text = text.replace('\n', ' <|newline|> ')
            text = text.replace('\t', ' <|tab|> ')
            words = text.split()
            for word in words:
                word_freqs[word] += 1
        return word_freqs

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def build_vocab(self, texts):
        word_freqs = self._get_word_freqs(texts)

        vocab = {}
        for word, freq in word_freqs.items():
            if word.startswith('<|') and word.endswith('|>'):
                vocab[word + ' </w>'] = freq
            else:
                vocab[' '.join(list(word)) + ' </w>'] = freq

        base_vocab = set()
        for word in vocab.keys():
            tokens = word.split()
            for token in tokens:
                if not (token.startswith('<|') and '|>' in token):
                    base_vocab.add(token)
                else:
                    base_vocab.add(token)

        vocab_tokens = sorted(list(base_vocab))

        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        available_for_merges = max(0, self.vocab_size - len(special_tokens) - len(vocab_tokens))

        for i in range(available_for_merges):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                word_pairs = self._get_pairs(word.split())
                for pair in word_pairs:
                    if not any(p.startswith('<|') and '|>' in p for p in pair):
                        pairs[pair] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

        all_tokens = special_tokens + vocab_tokens
        for pair in self.merges:
            all_tokens.append(''.join(pair))

        all_tokens = all_tokens[:self.vocab_size]

        self.token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)

    def _tokenize_word(self, word):
        if word.startswith('<|') and word.endswith('|>'):
            return [word, '</w>']

        word = word.split() + ['</w>']
        pairs = self._get_pairs(word)

        if not pairs:
            return word

        while True:
            bigram = min(pairs, key=lambda pair: self.merges.index(pair) if pair in self.merges else float('inf'))
            if bigram not in self.merges:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)

        return word

    def encode(self, text):
        text = text.replace('\n', ' <|newline|> ')
        text = text.replace('\t', ' <|tab|> ')
        words = text.split()
        tokens = []

        for word in words:
            word_tokens = self._tokenize_word(word if word.startswith('<|') and word.endswith('|>') else ' '.join(list(word)))
            for token in word_tokens:
                tokens.append(self.token_to_idx.get(token, self.token_to_idx['<unk>']))

        return tokens

    def decode(self, indices):
        tokens = [self.idx_to_token.get(idx, '<unk>') for idx in indices]

        result = []
        for i, token in enumerate(tokens):
            if token.startswith('<|') and token.endswith('|>'):
                if token == '<|newline|>':
                    result.append('\n')
                elif token == '<|tab|>':
                    result.append('\t')
                else:
                    result.append(token)
            else:
                if token.endswith('</w>'):
                    word_part = token[:-4]
                    result.append(word_part)
                    if i < len(tokens) - 1:
                        result.append(' ')
                else:
                    result.append(token)

        text = ''.join(result)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r' *\t *', '\t', text)
        text = text.strip()
        return text

    @property
    def eos_token_id(self):
        return self.token_to_idx['<eos>']
