# NumPyGPT

A PyTorch-inspired neural network library built from scratch using only NumPy. This is a pedagogical project focused on understanding how neural networks work by implementing the core components manually. No automatic differentiation, no tensor abstractions, just pure numpy arrays and explicit gradient computations.

## Philosophy

The best way to understand something is to build it yourself. This project implements neural network modules and optimizers using nothing but NumPy and Python's standard library.

Instead of relying on automatic differentiation, every layer explicitly implements its forward and backward passes. You can see exactly how gradients are computed and how they flow through the network. Every parameter update is visible, every mathematical operation is spelled out. The goal is clarity over performance.

## What This Is

- Neural network modules with explicit forward/backward implementations
- Educational tool for understanding backpropagation and optimization
- Test bed for experimenting with transformer architectures
- Clean, readable implementations of core deep learning algorithms
- PyTorch-like API without the complexity

## What This Isn't

- A full PyTorch clone (no tensors, no autograd engine)
- A production-ready framework (use PyTorch for that)
- Optimized for speed
- Feature-complete (only implements what's needed for transformer training)

## Project Structure

```
.
├── numpyGPT/                # Main package
│   ├── __init__.py
│   ├── nn/                  # Neural network modules
│   │   ├── __init__.py
│   │   ├── functional.py    # Stateless functions (e.g., F.cross_entropy)
│   │   ├── modules/         # Stateful modules
│   │   │   ├── __init__.py
│   │   │   ├── module.py    # Base Module class
│   │   │   ├── linear.py    # Linear layer with manual backward pass
│   │   │   ├── embedding.py # Token and positional embeddings
│   │   │   ├── layerNorm.py #  LayerNorm
│   │   └── init.py          # Parameter initialization
│   │
│   ├── optim/               # Optimizers
│   │   ├── __init__.py
│   │   ├── optimizer.py     # Base optimizer class
│   │   ├── adam.py          # AdamW optimizer implementation
│   │   └── lr_scheduler/    # Learning rate scheduling
│   │       ├── __init__.py
│   │       └── step_lr.py   # Simple step-based LR decay
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data/             # Data loading utilities
│   │       ├── __init__.py
│   │       ├── dataset.py    # Text dataset for character-level modeling
│   │       └── dataloader.py # Mini-batch generation
│   │
│   └── tokenizer/            # Text tokenization
│       ├── __init__.py
│       └── char_tokenizer.py  # Character-level tokenizer
│
├── tests/                   # Comprehensive testing suite
│   ├── __init__.py
│   ├── test_nn/             # Neural network component tests
│   │   ├── __init__.py
│   │   ├── test_functional.py
│   │   └── test_modules.py
│   ├── test_optim/          # Optimizer tests
│   │   ├── __init__.py
│   │   ├── test_adam.py
│   │   └── test_lr_scheduler.py
│   ├── test_utils/          # Utility tests
│   │   ├── __init__.py
│   │   └── test_data.py
│   ├── test_tokenizer.py
│   └── utils.py             # Gradient checking and test utilities
│
├── data/
│   └── shakespeare.txt      # Training data
│
├── models/
│   └── GPT.py
├── test.py
├── train.py
└── README.md
```