# 8.4 Training Deep Networks

## 🎯 Learning Objectives
After completing this section, you will master:
1. **Weight Initialization**: Xavier, He, LeCun initialization strategies
2. **Vanishing/Exploding Gradients**: Identify and solve gradient problems
3. **Batch Training**: Optimize batch size and memory usage
4. **Debugging Neural Networks**: Systematic approach to finding issues
5. **Gradient Flow Analysis**: Visualize and diagnose gradient problems

---

#### 🧒 ELI5: Weight Initialization, Vanishing/Exploding Gradients

> Imagine you're passing a message through a line of 100 people.
>
> **Weight Initialization** (Starting on the right foot):
>
> **Zero Initialization** (Bad start):
> - Everyone starts with SAME instruction: "Say nothing"
> - Person 1 whispers nothing → Person 2 hears nothing
> - ALL people learn the SAME thing!
> - Like: 100 clones, no diversity!
> - Network can't learn different features!
>
> **Random Initialization** (Better but tricky):
> - Each person gets RANDOM starting message
> - "Hello", "Cat", "Run", "Blue"...
> - Problem: Some whisper too quiet, some too loud!
>
> **Xavier Initialization** (Just right for tanh/sigmoid):
> - "Start with medium volume"
> - Not too quiet (vanishing)
> - Not too loud (exploding)
> - Goldilocks zone!
> - Formula: std = √(2 / (input + output))
>
> **He Initialization** (Just right for ReLU):
> - ReLU kills half the signals (negative → 0)
> - So start LOUDER to compensate!
> - Formula: std = √(2 / input)
> - Like: "Half of you will go silent, so others speak up!"
>
> **Vanishing Gradients** (Whisper game gone wrong):
>
> **The Problem**:
> - Person 1: "MEET AT NOON" (loud)
> - Person 10: "Meet at noon" (quieter)
> - Person 50: "meet at noon" (barely audible)
> - Person 100: "..." (silence)
>
> **What happens**:
> - Early people (layers 1-20): "Are we even learning?!"
> - Gradients become TINY (0.5¹⁰⁰ = 0.000000000000000000000000000001)
> - Weights don't update!
> - First layers NEVER learn!
>
> **Exploding Gradients** (Too loud!):
>
> **The Problem**:
> - Person 1: "meet" (quiet)
> - Person 10: "MEET!" (louder)
> - Person 50: "MEET!!!" (very loud)
> - Person 100: "📢📢📢MEET!!!📢📢📢" (EAR-BLEEDING)
>
> **What happens**:
> - Weights become HUGE (NaN!)
> - Network crashes!
> - Like: Microphone feedback screech!
>
> **Solutions**:
>
> **For Vanishing**:
> - Use ReLU (doesn't squash to zero)
> - Use LSTM (has memory highway)
> - Use Batch Norm (keeps scale consistent)
> - Use He/Xavier initialization (start right!)
>
> **For Exploding**:
> - Gradient Clipping ("Don't exceed volume 100!")
> - Use smaller learning rate
> - Use Batch Norm
> - Use LSTM/GRU (controlled gates)
>
> **Batch Training** (Learning in groups):
>
> **Batch Size** (How many examples before updating):
>
> **Batch = 1** (SGD):
> - See ONE example → update immediately
> - "This one is cat → update weights!"
> - "Next one is dog → update again!"
> - Fast updates, but JITTERY!
> - Like: Changing direction after every step
>
> **Batch = 1000** (Batch GD):
> - See 1000 examples → average → update once
> - Smooth, stable updates
> - But SLOW! Have to wait for 1000!
> - Like: Planning entire route before moving
>
> **Batch = 32** (Mini-batch - Sweet spot!):
> - See 32 examples → average → update
> - Fast AND stable!
> - Best of both worlds!
> - Like: Check map every few steps
>
> **Why mini-batch works**:
> - GPU parallelization (process 32 at once)
> - Good gradient estimate (not too noisy)
> - Regularization effect (some noise helps!)

</details>

---

## 📚 Weight Initialization

### 8.4.1 Why Initialization Matters

**Problem:** Poor initialization leads to:
- Vanishing/exploding gradients
- Slow convergence
- Getting stuck in poor local minima

**Goal:** Initialize weights so that:
- Activations don't vanish/explode
- Gradients flow properly
- Training starts in a good region

### 8.4.2 Common Initialization Methods

**1. Zero Initialization (DON'T USE)**
```python
# Problem: All neurons learn the same thing
W = np.zeros((n_out, n_in))  # ❌ Never do this!

# Why it fails:
# - All neurons have same output
# - Same gradients during backprop
# - Symmetry never broken
```

**2. Random Initialization**
```python
# Basic random initialization
W = np.random.randn(n_out, n_in) * 0.01  # Small random values

# Problem: Too small → vanishing gradients
#          Too large → exploding gradients
```

**3. Xavier/Glorot Initialization (for tanh, sigmoid)**
```python
def xavier_initialization(n_in, n_out):
    """
    Xavier initialization maintains variance across layers.
    Best for tanh and sigmoid activations.
    """
    # Xavier normal
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_out, n_in) * std
    
    # Xavier uniform (alternative)
    # limit = np.sqrt(6.0 / (n_in + n_out))
    # return np.random.uniform(-limit, limit, (n_out, n_in))
```

**Mathematical Justification:**
```
Goal: Var(y) = Var(x) where y = Wx

Var(y) = n_in · Var(W) · Var(x)

For Var(y) = Var(x):
Var(W) = 1 / n_in

Xavier uses: Var(W) = 2 / (n_in + n_out)
```

**4. He Initialization (for ReLU)**
```python
def he_initialization(n_in, n_out):
    """
    He initialization accounts for ReLU non-linearity.
    Recommended for ReLU and variants.
    """
    # He normal (recommended)
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_out, n_in) * std
    
    # He uniform (alternative)
    # limit = np.sqrt(6.0 / n_in)
    # return np.random.uniform(-limit, limit, (n_out, n_in))
```

**Why He for ReLU?**
```
ReLU kills half the neurons (negative values → 0)
This reduces variance by half

Compensation: Multiply variance by 2
Var(W) = 2 / n_in  (instead of 1 / n_in)
```

**5. LeCun Initialization (for SELU)**
```python
def leCun_initialization(n_in, n_out):
    """
    LeCun initialization for SELU activation.
    Used with self-normalizing networks.
    """
    std = np.sqrt(1.0 / n_in)
    return np.random.randn(n_out, n_in) * std
```

### Initialization Comparison

| Method | Formula | Best For | Avoid |
|--------|---------|----------|-------|
| **Zero** | W = 0 | Never | Always |
| **Random** | W ~ N(0, 0.01) | Quick tests | Production |
| **Xavier** | W ~ N(0, √(2/(n_in+n_out))) | tanh, sigmoid | ReLU |
| **He** | W ~ N(0, √(2/n_in)) | ReLU, Leaky ReLU | tanh, sigmoid |
| **LeCun** | W ~ N(0, √(1/n_in)) | SELU | Other activations |

### Implementation Example

```python
class WeightInitializer:
    """Weight initialization utilities"""
    
    @staticmethod
    def initialize(method, n_in, n_out, gain=1.0):
        """
        Initialize weights using specified method.
        
        Args:
            method: 'xavier', 'he', 'lecun', 'orthogonal', 'kaiming'
            n_in: Number of input units
            n_out: Number of output units
            gain: Scaling factor for weights
        """
        if method == 'xavier':
            std = gain * np.sqrt(2.0 / (n_in + n_out))
            return np.random.randn(n_out, n_in) * std
        
        elif method == 'he':
            std = gain * np.sqrt(2.0 / n_in)
            return np.random.randn(n_out, n_in) * std
        
        elif method == 'lecun':
            std = gain * np.sqrt(1.0 / n_in)
            return np.random.randn(n_out, n_in) * std
        
        elif method == 'orthogonal':
            # Orthogonal initialization (good for RNNs)
            a = np.random.randn(n_out, n_in)
            u, _, vt = np.linalg.svd(a, full_matrices=False)
            return gain * u if n_out > n_in else gain * vt
        
        elif method == 'kaiming_uniform':
            # Kaiming uniform (PyTorch default for ReLU)
            std = gain / np.sqrt(n_in)
            limit = np.sqrt(3.0) * std
            return np.random.uniform(-limit, limit, (n_out, n_in))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def get_gain(nonlinearity):
        """
        Get recommended gain for activation function.
        """
        gains = {
            'relu': np.sqrt(2),
            'leaky_relu': np.sqrt(2 / (1 + 0.01**2)),
            'tanh': 5.0 / 3,
            'sigmoid': 1,
            'linear': 1,
            'selu': 1
        }
        return gains.get(nonlinearity, 1.0)


# Example: Initialize a network
def initialize_network(layer_sizes, activation='relu', method='he'):
    """Initialize a multi-layer network"""
    parameters = {}
    
    for i in range(1, len(layer_sizes)):
        gain = WeightInitializer.get_gain(activation)
        parameters[f'W{i}'] = WeightInitializer.initialize(
            method, layer_sizes[i-1], layer_sizes[i], gain
        )
        parameters[f'b{i}'] = np.zeros((layer_sizes[i], 1))
    
    return parameters
```

---

## 📚 Vanishing/Exploding Gradients

### 8.4.3 Understanding the Problem

**Vanishing Gradients:**
```
Deep Network:
Input → Layer1 → Layer2 → Layer3 → Layer4 → Output

Backpropagation:
∂L/∂W1 = ∂L/∂output · ∂output/∂h4 · ∂h4/∂h3 · ∂h3/∂h2 · ∂h2/∂h1 · ∂h1/∂W1

If each ∂h_i+1/∂h_i < 1 (e.g., 0.5):
Product = 0.5 × 0.5 × 0.5 × 0.5 = 0.0625 → Vanishes!

Result: Early layers don't learn
```

**Exploding Gradients:**
```
If each ∂h_i+1/∂h_i > 1 (e.g., 2):
Product = 2 × 2 × 2 × 2 = 16 → Explodes!

Result: Weights become NaN, training diverges
```

**Visual Representation:**
```
Gradient Magnitude vs Layer Depth:

Good Flow:         Vanishing:        Exploding:
    │                  │                 │
    │                  │                 │
  ──┼──              ──┼──             ──┼──
    │                  │                ╱│
    │                 ╱│               ╱ │
    │                ╱ │              ╱  │
    │               ╱  │             ╱   │
    │              ╱   │            ╱    │
    └──────────    └──────────      └──────────
    Layer          Layer           Layer
```

### 8.4.4 Solutions

**1. Proper Initialization**
```python
# Use He initialization for ReLU networks
W = he_initialization(n_in, n_out)
```

**2. Batch Normalization**
```python
# Normalize activations to prevent extreme values
class BatchNorm:
    def forward(self, x):
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_norm = (x - mean) / np.sqrt(var + 1e-8)
        return self.gamma * x_norm + self.beta
```

**3. Gradient Clipping**
```python
def clip_gradients(grads, max_norm=1.0):
    """
    Clip gradients to prevent exploding.
    """
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        return [g * clip_coef for g in grads]
    return grads


# Usage in training
grads = compute_gradients(loss, params)
grads = clip_gradients(grads, max_norm=5.0)
update_parameters(params, grads, learning_rate)
```

**4. Residual Connections**
```python
# Residual block allows gradient to flow directly
def residual_block(x):
    identity = x  # Skip connection
    
    out = relu(conv1(x))
    out = relu(conv2(out))
    
    return out + identity  # Add input directly

# Gradient flows through both paths:
# ∂L/∂x = ∂L/∂out · (∂out/∂x + 1)
# The "+1" ensures gradient doesn't vanish
```

**5. Activation Function Choice**
```python
# Avoid sigmoid/tanh for deep networks
# Use ReLU or variants

def relu(x):
    return np.maximum(0, x)  # Gradient: 0 or 1

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)  # Never zero

def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

### Detecting Gradient Problems

```python
class GradientAnalyzer:
    """Analyze gradients during training"""
    
    def __init__(self, model):
        self.model = model
        self.gradient_history = []
    
    def analyze(self, grads):
        """
        Analyze gradient magnitudes.
        
        Returns dict with:
        - mean_magnitude: Average gradient magnitude
        - max_magnitude: Maximum gradient
        - min_magnitude: Minimum gradient
        - has_vanishing: Boolean
        - has_exploding: Boolean
        """
        grad_norms = [np.linalg.norm(g) for g in grads]
        
        mean_mag = np.mean(grad_norms)
        max_mag = np.max(grad_norms)
        min_mag = np.min(grad_norms)
        
        analysis = {
            'mean_magnitude': mean_mag,
            'max_magnitude': max_mag,
            'min_magnitude': min_mag,
            'has_vanishing': min_mag < 1e-7,
            'has_exploding': max_mag > 1000,
            'grad_norms': grad_norms
        }
        
        self.gradient_history.append(analysis)
        
        return analysis
    
    def plot_gradient_flow(self):
        """Visualize gradient magnitudes across layers"""
        import matplotlib.pyplot as plt
        
        if not self.gradient_history:
            print("No gradient history available")
            return
        
        latest = self.gradient_history[-1]
        
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(latest['grad_norms'])), latest['grad_norms'])
        plt.yscale('log')
        plt.xlabel('Layer')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Gradient Flow Across Layers')
        plt.axhline(y=1e-7, color='r', linestyle='--', label='Vanishing threshold')
        plt.axhline(y=1000, color='orange', linestyle='--', label='Exploding threshold')
        plt.legend()
        plt.show()
```

---

## 📚 Batch Training

### 8.4.5 Batch Size Selection

**Trade-offs:**
```
Small Batch (e.g., 16, 32):
✅ Better generalization
✅ Less memory usage
✅ More frequent updates
❌ Noisy gradients
❌ Slower training (less parallelization)

Large Batch (e.g., 256, 512):
✅ Stable gradients
✅ Faster training (better parallelization)
❌ Worse generalization
❌ More memory usage
❌ May converge to sharp minima
```

**Rules of Thumb:**
- Start with batch size 32 or 64
- Increase if training is stable
- Decrease if model overfits
- Use power of 2 for GPU efficiency

### 8.4.6 Gradient Accumulation

**Problem:** Model too large for GPU memory

**Solution:** Accumulate gradients over multiple mini-batches

```python
class GradientAccumulator:
    """
    Gradient accumulation for effective large batch training.
    """
    
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads = None
        self.step_counter = 0
    
    def accumulate(self, grads):
        """Accumulate gradients"""
        if self.accumulated_grads is None:
            self.accumulated_grads = [g.copy() for g in grads]
        else:
            for i, g in enumerate(grads):
                self.accumulated_grads[i] += g
        
        self.step_counter += 1
        
        # Return True when ready to update
        return self.step_counter >= self.accumulation_steps
    
    def get_accumulated_grads(self):
        """Get averaged gradients"""
        if self.accumulated_grads is None:
            return None
        
        # Average gradients
        avg_grads = [g / self.accumulation_steps for g in self.accumulated_grads]
        return avg_grads
    
    def reset(self):
        """Reset accumulator"""
        self.accumulated_grads = None
        self.step_counter = 0


# Training loop with gradient accumulation
def train_with_accumulation(model, data, accumulation_steps=4):
    accumulator = GradientAccumulator(accumulation_steps)
    
    for epoch in range(epochs):
        for i, (X_batch, y_batch) in enumerate(data):
            # Forward pass
            output = model(X_batch)
            loss = compute_loss(output, y_batch)
            
            # Backward pass
            grads = compute_gradients(loss, model.params)
            
            # Accumulate
            should_update = accumulator.accumulate(grads)
            
            if should_update:
                # Get averaged gradients
                avg_grads = accumulator.get_accumulated_grads()
                
                # Update parameters
                update_parameters(model.params, avg_grads, lr)
                
                # Reset accumulator
                accumulator.reset()
```

### 8.4.7 Mixed Precision Training

**Idea:** Use FP16 for computations, FP32 for master weights

```python
class MixedPrecisionTrainer:
    """
    Mixed precision training for faster computation.
    """
    
    def __init__(self, model, loss_scale=512.0):
        self.model = model
        self.loss_scale = loss_scale
        
        # Keep master weights in FP32
        self.master_weights = {k: v.astype(np.float32) 
                               for k, v in model.params.items()}
    
    def train_step(self, X, y):
        """Training step with mixed precision"""
        # Convert to FP16
        X_fp16 = X.astype(np.float16)
        
        # Forward pass (FP16)
        output = self.model.forward(X_fp16)
        loss = compute_loss(output, y.astype(np.float16))
        
        # Scale loss (prevent underflow)
        scaled_loss = loss * self.loss_scale
        
        # Backward pass
        grads = compute_gradients(scaled_loss, self.model.params)
        
        # Unscale gradients
        grads = [g / self.loss_scale for g in grads]
        
        # Update master weights (FP32)
        for k in self.master_weights:
            grad_k = grads[k].astype(np.float32)
            self.master_weights[k] -= self.lr * grad_k
        
        # Copy back to model (FP16)
        for k in self.model.params:
            self.model.params[k] = self.master_weights[k].astype(np.float16)
        
        return loss
```

---

## 📚 Debugging Neural Networks

### 8.4.8 Debugging Checklist

**1. Data Issues**
```python
def debug_data(X, y):
    """Check for common data issues"""
    print("=== Data Debugging ===")
    
    # Check for NaN/Inf
    print(f"NaN in X: {np.isnan(X).any()}")
    print(f"Inf in X: {np.isinf(X).any()}")
    print(f"NaN in y: {np.isnan(y).any()}")
    
    # Check data distribution
    print(f"X mean: {np.mean(X):.4f}")
    print(f"X std: {np.std(X):.4f}")
    print(f"X min: {np.min(X):.4f}")
    print(f"X max: {np.max(X):.4f}")
    
    # Check class balance
    if len(y.shape) == 1:
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Check for proper shuffling
    print(f"First 10 labels: {y[:10]}")
```

**2. Forward Pass Issues**
```python
def debug_forward(model, X):
    """Check forward pass outputs"""
    print("\n=== Forward Pass Debugging ===")
    
    output = model.forward(X)
    
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {np.mean(output):.4f}")
    print(f"Output std: {np.std(output):.4f}")
    print(f"Output NaN: {np.isnan(output).any()}")
    print(f"Output Inf: {np.isinf(output).any()}")
    
    # Check for dead neurons (ReLU)
    if hasattr(model, 'activations'):
        for i, act in enumerate(model.activations):
            dead_pct = np.mean(act == 0) * 100
            print(f"Layer {i} dead neurons: {dead_pct:.1f}%")
```

**3. Gradient Issues**
```python
def debug_gradients(model, X, y):
    """Check gradient flow"""
    print("\n=== Gradient Debugging ===")
    
    # Forward pass
    output = model.forward(X)
    loss = compute_loss(output, y)
    
    # Backward pass
    grads = model.backward(output, y)
    
    for i, grad in enumerate(grads):
        grad_norm = np.linalg.norm(grad)
        print(f"Layer {i} gradient norm: {grad_norm:.6f}")
        
        if np.isnan(grad).any():
            print(f"  ⚠️ NaN in gradient!")
        if np.isinf(grad).any():
            print(f"  ⚠️ Inf in gradient!")
        if grad_norm < 1e-7:
            print(f"  ⚠️ Vanishing gradient!")
        if grad_norm > 1000:
            print(f"  ⚠️ Exploding gradient!")
```

**4. Numerical Gradient Check**
```python
def numerical_gradient_check(model, X, y, epsilon=1e-5):
    """
    Verify analytical gradients using numerical approximation.
    """
    print("\n=== Numerical Gradient Check ===")
    
    # Get analytical gradients
    output = model.forward(X)
    analytical_grads = model.backward(output, y)
    
    max_diff = 0
    
    for param_name in model.params:
        param = model.params[param_name]
        numerical_grad = np.zeros_like(param)
        
        # Sample a few random indices
        indices = np.random.choice(param.size, min(10, param.size), replace=False)
        
        for idx in indices:
            # Convert flat index to multi-dimensional
            multi_idx = np.unravel_index(idx, param.shape)
            
            # f(x + epsilon)
            param[multi_idx] += epsilon
            output_plus = model.forward(X)
            loss_plus = compute_loss(output_plus, y)
            
            # f(x - epsilon)
            param[multi_idx] -= 2 * epsilon
            output_minus = model.forward(X)
            loss_minus = compute_loss(output_minus, y)
            
            # Restore
            param[multi_idx] += epsilon
            
            # Numerical gradient
            numerical_grad[multi_idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Compare
        analytical = analytical_grads[param_name].flatten()[indices]
        numerical = numerical_grad.flatten()[indices]
        
        diff = np.abs(analytical - numerical)
        max_diff = max(max_diff, np.max(diff))
        
        relative_error = diff / (np.abs(analytical) + np.abs(numerical) + 1e-8)
        print(f"{param_name} max relative error: {np.max(relative_error):.2e}")
    
    print(f"\nOverall max difference: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("✅ Gradients look correct!")
    elif max_diff < 1e-3:
        print("⚠️ Gradients might have issues")
    else:
        print("❌ Gradients are likely incorrect!")
```

### 8.4.9 Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Loss is NaN** | Loss becomes NaN immediately | Check for log(0), divide by 0, add epsilon |
| **Loss doesn't decrease** | Flat loss curve | Lower learning rate, check initialization |
| **Loss oscillates** | Wild swings in loss | Reduce learning rate, increase batch size |
| **Overfitting** | Train acc ↑, Val acc ↓ | Add regularization, more data, early stopping |
| **Underfitting** | Both acc low | Increase model capacity, train longer |
| **Slow convergence** | Very gradual improvement | Use better optimizer, normalize inputs |
| **Dead ReLU** | Many zero activations | Use Leaky ReLU, lower learning rate |

---

## 💻 Complete Training Pipeline with Debugging

```python
class DebuggableNeuralNetwork:
    """
    Neural network with comprehensive debugging tools.
    """
    
    def __init__(self, layer_sizes, activation='relu'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.params = {}
        self.cache = {}
        self.training_history = {
            'loss': [],
            'train_acc': [],
            'val_acc': [],
            'grad_norms': []
        }
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize with He initialization"""
        for i in range(1, len(self.layer_sizes)):
            n_in = self.layer_sizes[i-1]
            n_out = self.layer_sizes[i]
            
            # He initialization
            std = np.sqrt(2.0 / n_in)
            self.params[f'W{i}'] = np.random.randn(n_out, n_in) * std
            self.params[f'b{i}'] = np.zeros((n_out, 1))
    
    def forward(self, X):
        """Forward pass with caching for debugging"""
        self.cache['A0'] = X
        A = X
        
        L = len(self.params) // 2
        
        for l in range(1, L + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            A_prev = A
            
            Z = W @ A_prev + b
            self.cache[f'Z{l}'] = Z
            
            if l < L:
                # Hidden layer
                if self.activation == 'relu':
                    A = np.maximum(0, Z)
                elif self.activation == 'tanh':
                    A = np.tanh(Z)
                elif self.activation == 'sigmoid':
                    A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            else:
                # Output layer (sigmoid for binary classification)
                A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            
            self.cache[f'A{l}'] = A
        
        return A
    
    def backward(self, AL, Y):
        """Backward pass with gradient analysis"""
        grads = {}
        L = len(self.params) // 2
        m = AL.shape[1]
        
        # Output layer
        dAL = -(Y / (AL + 1e-15) - (1 - Y) / (1 - AL + 1e-15))
        Z = self.cache[f'Z{L}']
        dZ = dAL * (1 / (1 + np.exp(-Z))) * (1 - 1 / (1 + np.exp(-Z)))
        
        grads[f'dW{L}'] = (1 / m) * dZ @ self.cache[f'A{L-1}'].T
        grads[f'db{L}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers
        for l in reversed(range(1, L)):
            dA_prev = self.params[f'W{l+1}'].T @ dZ
            
            Z = self.cache[f'Z{l}']
            if self.activation == 'relu':
                dZ = dA_prev * (Z > 0)
            elif self.activation == 'tanh':
                dZ = dA_prev * (1 - np.tanh(Z) ** 2)
            elif self.activation == 'sigmoid':
                A = 1 / (1 + np.exp(-Z))
                dZ = dA_prev * A * (1 - A)
            
            grads[f'dW{l}'] = (1 / m) * dZ @ self.cache[f'A{l-1}'].T
            grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Store gradient norms for analysis
        grad_norms = [np.linalg.norm(grads[f'dW{l}']) for l in range(1, L + 1)]
        self.training_history['grad_norms'].append(grad_norms)
        
        return grads
    
    def train(self, X_train, Y_train, X_val, Y_val, 
              epochs=1000, learning_rate=0.01, print_cost=True):
        """Training loop with monitoring"""
        
        for epoch in range(epochs):
            # Forward
            AL = self.forward(X_train)
            
            # Compute loss
            cost = -np.mean(Y_train * np.log(AL + 1e-15) + 
                           (1 - Y_train) * np.log(1 - AL + 1e-15))
            
            # Backward
            grads = self.backward(AL, Y_train)
            
            # Update
            for l in range(1, len(self.params) // 2 + 1):
                self.params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
                self.params[f'b{l}'] -= learning_rate * grads[f'db{l}']
            
            # Record history
            self.training_history['loss'].append(cost)
            
            # Compute accuracies
            train_pred = (self.forward(X_train) > 0.5).astype(int)
            val_pred = (self.forward(X_val) > 0.5).astype(int)
            
            train_acc = np.mean(train_pred == Y_train)
            val_acc = np.mean(val_pred == Y_val)
            
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            if print_cost and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {cost:.4f}, "
                      f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        
        return self
    
    def plot_training_history(self):
        """Plot training curves"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss curve
        axes[0].plot(self.training_history['loss'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)
        
        # Accuracy curves
        axes[1].plot(self.training_history['train_acc'], label='Train')
        axes[1].plot(self.training_history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
```

---

## 📊 Summary Tables

### Initialization Methods

| Method | Variance | Use Case | Activation |
|--------|----------|----------|------------|
| Xavier | 2/(n_in + n_out) | Balanced flow | tanh, sigmoid |
| He | 2/n_in | ReLU networks | ReLU, Leaky ReLU |
| LeCun | 1/n_in | Self-normalizing | SELU |
| Orthogonal | - | RNNs, stability | Any |

### Gradient Problems

| Problem | Cause | Detection | Solution |
|---------|-------|-----------|----------|
| Vanishing | Deep networks, sigmoid | Grad norm < 1e-7 | ReLU, BatchNorm, residuals |
| Exploding | Large weights, deep | Grad norm > 1000 | Clipping, BatchNorm |
| Dead ReLU | Negative inputs | Many zero activations | Leaky ReLU, lower LR |

### Debugging Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| Gradient check | Verify backprop | Initial implementation |
| Activation stats | Detect dead neurons | Training issues |
| Loss monitoring | Track convergence | Every training run |
| Data validation | Catch data issues | Before training |

---

## 📝 Practice Problems

### Level 1: Basic
1. Explain why zero initialization fails
2. Calculate He initialization std for layer with 256 inputs
3. Identify vanishing vs exploding gradient symptoms
4. Implement gradient clipping function
5. List 3 causes of NaN loss

### Level 2: Intermediate
1. Compare Xavier and He initialization experimentally
2. Implement gradient accumulation for large batch training
3. Build a gradient norm visualizer
4. Debug a network with dead ReLU problem
5. Implement numerical gradient checking

### Level 3: Advanced
1. Implement mixed precision training from scratch
2. Build comprehensive debugging framework
3. Analyze gradient flow in very deep networks (100+ layers)
4. Implement automatic debugging that suggests fixes
5. Research and implement advanced initialization (e.g., FixUp)

---

**Status:** ✅ Complete  
**Next:** [[05-Convolutional-Neural-Networks]]
