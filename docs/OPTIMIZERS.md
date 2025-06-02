# Optimizers: How Neural Networks Learn

*Turning gradients into better weights*

## The Problem

You have gradients. They point in the direction of steepest increase. You want to minimize loss. So you move in the opposite direction. But how far? How fast? 

Naive approach: `weights -= learning_rate * gradients`

This works, but slowly. Modern optimizers are smarter.

## SGD: The Foundation

```python
class SGD:
    def __init__(self, modules, lr=0.01):
        self.params = modules
        self.lr = lr
        
    def step(self):
        for module in self.params:
            params = module.params()
            grads = module.grads()
            for param_key in params:
                params[param_key] -= self.lr * grads[param_key]
```

**The update rule:** `θ = θ - lr * ∇θ`

**Problems:**
- Same learning rate for all parameters
- No memory of previous gradients  
- Gets stuck in narrow valleys
- Oscillates around optimal points

## Adam: Adaptive Moments

Adam tracks two moving averages:
- **Momentum** (first moment): direction of gradients
- **RMSprop** (second moment): magnitude of gradients

```python
class Adam:
    def __init__(self, modules, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        self.m = []  # momentum
        self.v = []  # velocity (squared gradients)
        
    def step(self):
        self.t += 1
        
        for i, module in enumerate(self.params):
            grads = module.grads()
            for param_key in grads:
                g = grads[param_key]
                
                # Update biased first moment estimate
                self.m[i][param_key] = self.beta1 * self.m[i][param_key] + (1 - self.beta1) * g
                
                # Update biased second moment estimate  
                self.v[i][param_key] = self.beta2 * self.v[i][param_key] + (1 - self.beta2) * g**2
                
                # Bias correction
                m_hat = self.m[i][param_key] / (1 - self.beta1**self.t)
                v_hat = self.v[i][param_key] / (1 - self.beta2**self.t)
                
                # Update parameters
                params[param_key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

## Why Adam Works

**Momentum (`m`):** Smooths gradients. Builds velocity in consistent directions.
```python
m = β₁ * m + (1-β₁) * g    # β₁=0.9 typical
```

**RMSprop (`v`):** Adapts learning rate per parameter. Large gradients → smaller steps.
```python  
v = β₂ * v + (1-β₂) * g²   # β₂=0.999 typical
```

**Bias correction:** Early iterations have small `m` and `v`. Correction compensates.
```python
m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)
```

**Final update:** Momentum direction, RMSprop magnitude.
```python
θ = θ - lr * m̂ / (√v̂ + ε)
```

## The Intuition

**SGD:** Walk downhill with fixed step size.

**Adam:** 
- Walk in the average direction you've been going (momentum)
- Take smaller steps where the path is steep (adaptive learning rate)
- Correct for startup effects (bias correction)

## Hyperparameters

**Learning rate (`lr`):** How big steps to take
- Too high: overshoots, unstable
- Too low: slow convergence
- Typical: 1e-3 to 3e-4

**Beta1 (`β₁`):** Momentum decay  
- Higher = more momentum
- Typical: 0.9

**Beta2 (`β₂`):** RMSprop decay
- Higher = longer memory of gradient magnitudes  
- Typical: 0.999

**Epsilon (`ε`):** Numerical stability
- Prevents division by zero
- Typical: 1e-8

## Learning Rate Scheduling

Fixed learning rates are suboptimal. Common schedules:

**Warmup + Cosine Decay:**
```python
if step < warmup_steps:
    lr = max_lr * step / warmup_steps
elif step > decay_steps:
    lr = min_lr  
else:
    progress = (step - warmup_steps) / (decay_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
```

**Step Decay:**
```python
lr = base_lr * gamma ** (step // step_size)
```

## Comparison

| Optimizer | Memory | Convergence | Hyperparams | Use Case |
|-----------|---------|-------------|-------------|----------|
| SGD       | None    | Slow, noisy | 1 (lr)      | Simple problems |
| Adam      | 2x params | Fast, stable | 4 (lr, β₁, β₂, ε) | Most deep learning |

## Implementation Pattern

All optimizers follow this interface:

```python
class Optimizer:
    def __init__(self, modules, lr):
        self.params = modules
        self.lr = lr
        
    def step(self):
        # Update parameters using gradients
        raise NotImplementedError
        
    def zero_grad(self):
        # Reset gradients to zero to avoid contamination of the next batch
        for module in self.params:
            grads = module.grads()
            for grad_key in grads:
                grads[grad_key].fill(0.0)
```

The optimizer determines how quickly your model learns. SGD is simple but slow. Adam is adaptive and fast. For modern deep learning, Adam with learning rate scheduling is the default choice. 