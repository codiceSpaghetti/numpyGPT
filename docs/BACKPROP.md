# Backpropagation: A Hands-On Guide

*The algorithm that makes neural networks learn*

## The Problem

You have a neural network. It makes predictions. They're wrong. You need to adjust the weights to make better predictions. But there are millions of weights. Which ones to change? By how much?

Naive approach: try every weight, see what happens. This takes forever.

Smart approach: compute gradients. This tells you exactly how each weight affects the loss. One backward pass gives you all the information you need.

## The Solution: Chain Rule

```python
# If f(x) = h(g(x)), then:
# df/dx = (dh/dg) * (dg/dx)
```

That's it. Everything else is just applying this rule systematically.

## Linear Layer Implementation

```python
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.02
        self.b = np.zeros(out_features)
    
    def forward(self, X):
        self.X = X  # cache for backward
        return X @ self.W + self.b
    
    def backward(self, dY):
        # dY: gradient flowing back from next layer
        self.dW = self.X.T @ dY        # gradient w.r.t weights
        self.db = np.sum(dY, axis=0)   # gradient w.r.t bias  
        dX = dY @ self.W.T             # gradient w.r.t input
        return dX
```

## Why These Formulas Work

Forward: `Y = X @ W + b`

Backward: Given `dY` (how loss changes w.r.t Y), find how loss changes w.r.t X, W, b.

**Weight gradient:** `dW = X.T @ dY`
- From Y = XW, we get ∂Y/∂W = X  
- Chain rule: ∂L/∂W = ∂L/∂Y × ∂Y/∂W = dY × X
- Matrix dimensions: need X.T @ dY to get correct shape

**Bias gradient:** `db = sum(dY, axis=0)`
- From Y = XW + b, we get ∂Y/∂b = 1
- Chain rule: ∂L/∂b = ∂L/∂Y × 1 = dY
- Sum over batch since bias is shared

**Input gradient:** `dX = dY @ W.T` 
- From Y = XW, we get ∂Y/∂X = W
- Chain rule: ∂L/∂X = ∂L/∂Y × ∂Y/∂X = dY × W
- Matrix dimensions: need dY @ W.T to get correct shape

## Example Walkthrough

```python
# Setup
X = np.array([[1, 2], [3, 4]])  # batch_size=2, in_features=2
W = np.array([[0.1, 0.3], [0.2, 0.4]])  # 2x2 weight matrix
b = np.array([0.1, 0.2])

# Forward
Y = X @ W + b
# Y = [[0.6, 1.3], [1.4, 2.7]]

# Backward (assume dY = ones)
dY = np.ones_like(Y)  # [[1, 1], [1, 1]]

dW = X.T @ dY  # [[4, 4], [6, 6]]
db = np.sum(dY, axis=0)  # [2, 2] 
dX = dY @ W.T  # [[0.4, 0.6], [0.4, 0.6]]
```

## Other Common Layers

**ReLU:**
```python
def relu_forward(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    return dout * (x > 0)
```

**Softmax + CrossEntropy:**
```python
def softmax_crossentropy_backward(probs, targets):
    batch_size = probs.shape[0]
    dlogits = probs.copy()
    dlogits[range(batch_size), targets] -= 1
    return dlogits / batch_size
```

## Implementation Pattern

Every layer follows this pattern:

```python
class Layer:
    def forward(self, x):
        # 1. cache inputs needed for backward
        # 2. compute output
        # 3. return output
        
    def backward(self, dout):
        # 1. use cached values
        # 2. compute parameter gradients (self.dW, self.db, etc.)
        # 3. compute input gradients
        # 4. return input gradients
```

## Gradient Checking

Always test your backward pass:

```python
def gradient_check(f, x, analytic_grad):
    h = 1e-7
    grad_numerical = (f(x + h) - f(x - h)) / (2 * h)
    rel_error = abs(analytic_grad - grad_numerical) / (abs(analytic_grad) + abs(grad_numerical))
    assert rel_error < 1e-7, f"gradient check failed: {rel_error}"
```

## Shape Debugging

Most bugs are shape mismatches. Always check:

```python
print(f"X: {X.shape}, W: {W.shape}, Y: {Y.shape}")
print(f"dY: {dY.shape}, dW: {dW.shape}, dX: {dX.shape}")
```

The gradient of any variable must have the same shape as the variable itself.

## The Training Loop

```python
# forward pass
y_pred = model(x)
loss = loss_fn(y_pred, y_true)

# backward pass  
model.zero_grad()
loss.backward()

# update weights
optimizer.step()
```

That's backpropagation. It's just the chain rule applied systematically to compute gradients efficiently. Once you understand the math and can implement a few layers, you understand the engine that powers all of deep learning. 