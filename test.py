import os
import sys
import unittest

import numpy as np


def run_tests():
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    loader = unittest.TestLoader()
    start_dir = os.path.join(project_root, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def demo_dataloader():
    import pickle
    import tempfile

    from numpyGPT.utils.data import DataLoader

    print("=== DataLoader Demo ===")

    temp_dir = tempfile.mkdtemp()

    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    text = "hello world this is a test of the dataloader component"
    data = np.array([stoi[c] for c in text], dtype=np.uint16)

    data.tofile(os.path.join(temp_dir, 'train.bin'))

    meta = {'vocab_size': len(vocab), 'stoi': stoi, 'itos': itos}
    with open(os.path.join(temp_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    loader = DataLoader(temp_dir, 'train', batch_size=2, block_size=8)

    print(f"Vocab size: {loader.vocab_size}")
    print(f"Data length: {len(loader.data)}")

    X, y = loader.get_batch()
    print(f"Batch shapes: X={X.shape}, y={y.shape}")

    for i in range(X.shape[0]):
        x_text = loader.decode(X[i])
        y_text = loader.decode(y[i])
        print(f"Sample {i}: '{x_text}' -> '{y_text}'")


def demo_adam_optimizer():
    from numpyGPT.nn.modules import Linear
    from numpyGPT.optim import Adam

    print("=== Adam Optimizer Demo ===")
    np.random.seed(42)

    layer = Linear(2, 1)
    optimizer = Adam([layer], lr=0.01)

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_true = np.array([[3.0], [7.0], [11.0]])

    print(f"Initial weights: W={layer.W.flatten()}, b={layer.b}")

    for epoch in range(100):
        y_pred = layer.forward(X)
        loss = np.mean((y_pred - y_true) ** 2)

        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")

        dY = 2 * (y_pred - y_true) / len(X)
        layer.backward(dY)

        optimizer.step()
        optimizer.zero_grad()

    print(f"Final weights: W={layer.W.flatten()}, b={layer.b}")
    print(f"Final loss: {loss:.6f}")
    print("Expected: W≈[1, 1], b≈[1] for linear relation y = x1 + x2 + 1")


if __name__ == '__main__':
    demo_dataloader()
    print("\n" + "="*50 + "\n")

    demo_adam_optimizer()
    print("\n" + "="*50 + "\n")

    success = run_tests()
    sys.exit(0 if success else 1)
