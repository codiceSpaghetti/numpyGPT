# Tokenizers: From Text to Numbers

*How neural networks understand language.*

## the problem

Neural networks work with numbers. Text is strings. You need to convert "Hello world" into `[1, 42, 7]` somehow. But which conversion? 

* Character-level: `['H','e','l','l','o',' ','w','o','r','l','d']`
* Word-level: `['Hello', 'world']`  
* BPE: `['Hel', 'lo', 'world']`

Each choice shapes what your model learns.

## character-level: simple and universal

```python
class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        
    def build_vocab(self, texts):
        chars = set(''.join(texts))
        for i, char in enumerate(sorted(chars)):
            self.stoi[char] = i
            self.itos[i] = char
            
    def encode(self, text):
        return [self.stoi[char] for char in text]
        
    def decode(self, indices):
        return ''.join(self.itos[i] for i in indices)
```

**Pros:** Never sees unknown words. Works for any language.<br>
**Cons:** Long sequences. Hard to learn word meanings.

## word-level: semantic units

```python
class WordTokenizer:
    def tokenize(self, text):
        return re.findall(r"\b\w+\b|[^\s\w]", text.lower())
        
    def build_vocab(self, texts):
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
            
        counts = Counter(all_tokens)
        vocab = [token for token, count in counts.most_common() 
                if count >= self.min_freq]
```

**Pros:** Captures word meanings. Shorter sequences.<br>
**Cons:** Large vocabularies. Unknown words become `<unk>`.

## bpe: best of both worlds

Start with characters. Merge frequent pairs iteratively.

```python
# Initial: "hello" -> ['h', 'e', 'l', 'l', 'o']
# Count pairs: ('l','l') appears most
# Merge: "hello" -> ['h', 'e', 'll', 'o'] 
# Repeat until vocab_size reached
```

The algorithm:

```python
def build_vocab(self, texts):
    # Start with character-level splits
    vocab = {}
    for word, freq in word_freqs.items():
        vocab[' '.join(word) + ' </w>'] = freq
    
    # Iteratively merge most frequent pairs
    for _ in range(num_merges):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            word_pairs = get_pairs(word.split())
            for pair in word_pairs:
                pairs[pair] += freq
                
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        self.merges.append(best_pair)
```

**Example walkthrough:**
```
Text: "hello hello world"
Initial vocab: {'h e l l o </w>': 2, 'w o r l d </w>': 1}

Iteration 1: Most frequent pair is ('l', 'l')
Merge: {'h e ll o </w>': 2, 'w o r l d </w>': 1}

Iteration 2: Most frequent pair is ('h', 'e') 
Merge: {'he ll o </w>': 2, 'w o r l d </w>': 1}

Continue until vocab_size reached...
```

## the three approaches compared

| Method | Sequence Length | Unknown Words | Use Case |
|--------|-----------------|---------------|----------|
| Char   | Very Long       | Never         | Small data, multilingual |
| Word   | Short           | Common        | Large vocab, semantic tasks |
| BPE    | Medium          | Rare          | Production systems |

## special tokens

Every tokenizer needs these:

```python
special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
# <pad>: padding for batches
# <unk>: unknown words  
# <bos>: beginning of sequence
# <eos>: end of sequence
```

## implementation pattern

All tokenizers follow this interface:

```python
class Tokenizer:
    def build_vocab(self, texts):
        # Build token <-> index mappings
        pass
        
    def encode(self, text):
        # text -> list of indices
        return [self.stoi[token] for token in tokens]
        
    def decode(self, indices):
        # indices -> text (join method depends on tokenizer type)
        tokens = [self.itos[i] for i in indices]
        return ''.join(tokens)  # char-level: no spaces
        # return ' '.join(tokens)  # word-level: spaces between words
```
