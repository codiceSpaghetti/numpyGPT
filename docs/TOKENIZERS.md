# Tokenizers: From Text to Numbers

*How neural networks understand language*

## The Problem

Neural networks work with numbers. Text is strings. You need to convert "Hello world" into `[1, 42, 7]` somehow. But which conversion? 

Character-level: `['H','e','l','l','o',' ','w','o','r','l','d']`
Word-level: `['Hello', 'world']`  
BPE: `['Hel', 'lo', 'world']`

Each choice shapes what your model learns.

## Character-Level: Simple and Universal

```python
class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        
    def build_vocab(self, texts):
        chars = set(''.join(texts))
        for i, char in enumerate(sorted(chars)):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
            
    def encode(self, text):
        return [self.char_to_idx[char] for char in text]
        
    def decode(self, indices):
        return ''.join(self.idx_to_char[i] for i in indices)
```

**Pros:** Never sees unknown words. Works for any language.
**Cons:** Long sequences. Hard to learn word meanings.

## Word-Level: Semantic Units

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

**Pros:** Captures word meanings. Shorter sequences.
**Cons:** Large vocabularies. Unknown words become `<unk>`.

## BPE: Best of Both Worlds

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

## The Three Approaches Compared

| Method | Vocab Size | Sequence Length | Unknown Words | Use Case |
|--------|------------|-----------------|---------------|----------|
| Char   | ~100       | Very Long       | Never         | Small data, multilingual |
| Word   | ~50,000    | Short           | Common        | Large vocab, semantic tasks |
| BPE    | ~1,000     | Medium          | Rare          | Production systems |

## Special Tokens

Every tokenizer needs these:

```python
special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
# <pad>: padding for batches
# <unk>: unknown words  
# <bos>: beginning of sequence
# <eos>: end of sequence
```

## Implementation Pattern

All tokenizers follow this interface:

```python
class Tokenizer:
    def build_vocab(self, texts):
        # Build token <-> index mappings
        pass
        
    def encode(self, text):
        # text -> list of indices
        return [self.token_to_idx[token] for token in tokens]
        
    def decode(self, indices):
        # indices -> text
        return ''.join(self.idx_to_token[i] for i in indices)
```

## Choosing Your Tokenizer

**Start with char-level** if:
- Small dataset (< 1M tokens)
- Multiple languages
- Lots of rare words

**Use word-level** if:
- Large vocabulary is fine
- Semantic understanding matters
- English-only

**Use BPE** if:
- Production system
- Memory/compute constraints matter
- Need balance of all factors

The tokenizer determines your model's view of language. Character-level sees letters. Word-level sees concepts. BPE sees frequent patterns. Choose the view that matches your task. 