# PyTorch Comparison Tests

Tests comparing numpyGPT components with PyTorch equivalents.

## Files

- `test_components.py` - Basic component tests (Linear, Embedding, LayerNorm, ReLU, Softmax, FeedForward, Attention, Adam)
- `test_blocks.py` - Higher-level block tests (PositionalEncoding, TransformerBlock, GPT model, CrossEntropy loss)

## Running Tests

```bash
python -m unittest tests.pytorch.test_components -v
python -m unittest tests.pytorch.test_blocks -v
```

Or all tests:
```bash
python -m unittest discover tests/pytorch -v
``` 