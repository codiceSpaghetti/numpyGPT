from numpyGPT.nn.modules.activation import Softmax
import numpy as np

from numpyGPT.models.GPT import GPT
from numpyGPT.nn.modules.activation import LeakyReLU, ReLU

print("=== NumPyGPT Demo ===")

# Model hyperparameters
vocab_size = 50
max_len = 32
d_model = 128
n_heads = 8
n_layers = 4
d_ff = 512

print(f"Creating GPT model with:")
print(f"  vocab_size: {vocab_size}")
print(f"  max_len: {max_len}")
print(f"  d_model: {d_model}")
print(f"  n_heads: {n_heads}")
print(f"  n_layers: {n_layers}")
print(f"  d_ff: {d_ff}")

print("\n=== Testing ReLU Model ===")
model = GPT(vocab_size, max_len, d_model, n_heads, n_layers, d_ff)

# Create sample data
B, T = 2, 16
X = np.random.randint(0, vocab_size, (B, T))
targets = np.random.randint(0, vocab_size, (B, T))

print(f"\nInput shape: {X.shape}")
print(f"Targets shape: {targets.shape}")

print(f"\n=== ReLU Forward Pass ===")
logits, loss = model(X, targets)
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss:.4f}")

print(f"\n=== ReLU Backward Pass ===")
model.backward()

params = model.params()
grads = model.grads()
print(f"Number of parameters: {len(params)}")
print(f"Number of gradients: {len(grads)}")

total_params = sum(p.size for p in params.values())
print(f"Total parameters: {total_params:,}")

print(f"\n=== ReLU Generation Demo ===")
start_tokens = np.array([[1, 2, 3]])
print(f"Start tokens: {start_tokens[0]}")

generated = model.generate(start_tokens, max_new_tokens=10)
print(f"Generated sequence: {generated[0]}")

print(f"\n=== Softmax Module Reuse Demo ===")

softmax = Softmax()
X_test = np.random.randn(3, 4)
Y_true = np.array([0, 2, 1])

# Test classification backward (original functionality)
probs = softmax(X_test)
grad_classification = softmax.backward(probs, Y_true)
print("Testing classification backward (original functionality):")
print(f"Classification gradient shape: {grad_classification.shape}")

# Test general backward (new functionality for attention)
dZ = np.random.randn(3, 4)
probs = softmax(X_test)
grad_general = softmax.backward(dZ)
print("Testing general backward (new functionality for attention):")
print(f"General gradient shape: {grad_general.shape}")

print(f"\n=== ReLU vs LeakyReLU Comparison ===")
relu = ReLU()
leaky_relu = LeakyReLU(alpha=0.01)

X_test = np.random.randn(2, 4, 8)
print(f"Input shape: {X_test.shape}")

relu_out = relu(X_test)
leaky_relu_out = leaky_relu(X_test)

print(f"ReLU output shape: {relu_out.shape}")
print(f"LeakyReLU output shape: {leaky_relu_out.shape}")

print("\nDemo completed successfully!")
