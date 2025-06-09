# backpropagation: intuition

*The algorithm that makes neural networks learn.*

## the problem

You have a neural network. It makes predictions. They're wrong. You need to adjust the weights to make better predictions. But there are millions of weights. Which ones to change? By how much?

You just need one basic concept from calculus: **derivatives**.
A derivative tells you how a small change in something (a weight) affects something else (the **loss**).

Once you have those gradients, you do something simple: move each weight a little bit in the **opposite direction of the gradient**.
That’s called **gradient descent**.
You’re climbing down the **loss curve**, looking for the lowest point.

How to compute gradients?

We split the problem into smaller parts and use the **chain rule** to compute the gradient of the loss with respect to each part.

### why 'backprop'?

We **propagate the error backward** through the network. Starting from the output's loss, we apply the **chain rule layer by layer**, working our way back to the inputs.
At each step, we compute how much each node contributed to the error, and how much its weights need to change to reduce it.

Each layer passes its gradients to the one before it, so there is a **top-down flow of information** to provide to every parameter its update direction.

That’s it. Backprop is just an **efficient application of the chain rule** to compute gradients in a big function (e.g., a transformer) composed of many smaller ones.


## the solution: chain rule

```python
# If f(x) = h(g(x)), then:
# df/dx = (dh/dg) * (dg/dx)
```

That's all. Everything else is just applying this rule systematically.

## linear layer implementation

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

## why these formulas work

Forward: `Y = X @ W + b`

Backward: Given `dY` (how loss changes w.r.t Y), find how loss changes w.r.t X, W, b.

**Weight gradient:** `dW = X.T @ dY`
- From Y = XW, we get ∂Y/∂W = X  
- Chain rule: ∂L/∂W = ∂L/∂Y × ∂Y/∂W = dY × X

**Bias gradient:** `db = sum(dY, axis=0)`
- From Y = XW + b, we get ∂Y/∂b = 1
- Chain rule: ∂L/∂b = ∂L/∂Y × 1 = dY

**Input gradient:** `dX = dY @ W.T` 
- From Y = XW, we get ∂Y/∂X = W
- Chain rule: ∂L/∂X = ∂L/∂Y × ∂Y/∂X = dY × W

## example walkthrough

```python
# Setup
X = np.array([[1, 2], [3, 4]])  # batch_size=2, in_features=2
W = np.array([[0.1, 0.3], [0.2, 0.4]])  # 2x2 weight matrix
b = np.array([0.1, 0.2])

# Forward
Y = X @ W + b
# Y = [[0.6, 1.3], [1.2, 2.7]]

# Backward (assume dY = ones)
dY = np.ones_like(Y)  # [[1, 1], [1, 1]]

dW = X.T @ dY  # [[4, 4], [6, 6]]
db = np.sum(dY, axis=0)  # [2, 2] 
dX = dY @ W.T  # [[0.4, 0.6], [0.4, 0.6]]
```

## other common layers

**ReLU:**
```python
def relu_forward(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    return dout * (x > 0)
```

## implementation pattern

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


## shape debugging

Most bugs are shape mismatches. Always check:

```python
print(f"X: {X.shape}, W: {W.shape}, Y: {Y.shape}")
print(f"dY: {dY.shape}, dW: {dW.shape}, dX: {dX.shape}")
```

The gradient of any variable must have the same shape as the variable itself.

## the training loop

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