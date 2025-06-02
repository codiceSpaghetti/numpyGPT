# NumPyGPT

Neural networks from scratch. Just NumPy and Python.

## Why?

Understanding comes from building. This implements the core pieces of neural networks - modules, tokenizers, optimizers, backpropagation - using only NumPy. No autograd, no tensor abstractions. Every gradient computation is explicit.

**Core modules:**
- `Linear`, `Embedding`, `LayerNorm`, `Softmax`, `ReLU`
- `Adam` optimizer with learning rate scheduling  
- `GPT` model (transformer decoder)
- Character-level, word-level, and BPE tokenizers

**Educational resources:**
- [BACKPROP.md](docs/BACKPROP.md) - implement backward passes from scratch
- [OPTIMIZERS.md](docs/OPTIMIZERS.md) - understand the difference between Adam and SGD
- [TOKENIZERS.md](docs/TOKENIZERS.md) - understand the difference between character-level, word-level, and BPE tokenization

## Quick Start

```bash
# Download training data
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

pip install numpy 

# Character-level tokenization (default), but are also supported word-level and BPE tokenization
python datagen.py

# BPE tokenization
python datagen.py input.txt --output_dir data/shakespeare_bpe --tokenizer_type bpe --vocab_size 1000

# Train the model
python train.py

# Generate some text
python sample.py --num_samples 3 --temperature 0.8

# Plot training curves
python plot.py --out_dir out/char
```

## Tokenization Options

```bash
# Character-level (default)
python datagen.py input.txt --output_dir data/shakespeare_char

# Word-level with vocabulary constraints
python datagen.py input.txt --output_dir data/shakespeare_word \
  --tokenizer_type word \
  --min_freq 2 \
  --max_vocab_size 10000

# BPE (Byte-Pair Encoding)
python datagen.py input.txt --output_dir data/shakespeare_bpe \
  --tokenizer_type bpe \
  --vocab_size 1000
```

## Implementation Example

```python
# Every layer follows this pattern
class Linear:
    def forward(self, x):
        self.x = x  # cache for backward
        return x @ self.W + self.b
    
    def backward(self, dy):
        self.dW = self.x.T @ dy
        self.db = np.sum(dy, axis=0) 
        return dy @ self.W.T

# Training loop
for batch in dataloader:
    logits, loss = model(x, y)
    model.backward()
    optimizer.step()
```

## Project Structure

```
numpyGPT/
├── nn/
│   ├── modules/          # Linear, Embedding, LayerNorm, etc.
│   └── functional.py     # cross_entropy, softmax, etc.
├── optim/               # Adam optimizer + LR scheduling
├── utils/data/          # DataLoader, Dataset
├── tokenizer/           # Character, word-level & BPE tokenizers
└── models/GPT.py        # Transformer implementation

datagen.py              # Data preprocessing
train.py                # Training script
sample.py               # Text generation
plot.py                 # Training curves
test.py                 # Test suite
```

## Features

- **Explicit gradients** - see exactly how backprop works
- **PyTorch-like API** - familiar interface, educational internals  
- **Complete transformer** - multi-head attention, feedforward, layer norm
- **Flexible tokenization** - character, word-level, or BPE preprocessing
- **Gradient checking** - verify your backward implementations
- **Minimal dependencies** - just numpy and standard library

Perfect for understanding how modern language models actually work.

## References
A few resources that helped me build this:
- [Pytorch](https://github.com/pytorch/pytorch) - architecture and API inspiration
- [LayerNorm in llm.c (Karpathy's again <3)](https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md) - clean example of LayerNorm
- [CNN in Numpy for MNIST](https://github.com/ScottBiggs2/Generative-AI-Projects/blob/main/AI%20in%20Numpy/NNs%20from%20Scratch%20-%20Clean.ipynb) - end-to-end CNN in NumPy
- [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&ab_channel=AndrejKarpathy) - autograd from scratch, explained simply
- [Kaggle's L-layer neural network using numpy](https://www.kaggle.com/code/mtax687/l-layer-neural-network-using-numpy) - some NumPy-based deep nets
- [Forward and Backpropagation in Neural Networks](https://github.com/xbeat/Machine-Learning/blob/main/Forward%20and%20Backpropagation%20in%20Neural%20Networks%20using%20Python.md) - to understand the approach to implement forward and backward pass