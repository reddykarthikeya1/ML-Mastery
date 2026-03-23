# 8.3 Regularization Techniques

## 🎯 Learning Objectives
After completing this section, you will master:
1. **L1/L2 Regularization**: Understand weight decay and sparsity
2. **Dropout**: Master dropout variants and implementation
3. **Batch Normalization**: Understand internal covariate shift and normalization
4. **Advanced Normalization**: LayerNorm, InstanceNorm, GroupNorm
5. **Other Techniques**: Early stopping, data augmentation, label smoothing, and more

---

## 📚 Regularization Fundamentals

### What is Regularization?

**Definition:** Techniques that prevent overfitting by constraining the learning process

**Goal:**
```
Without Regularization:        With Regularization:
   High Training Acc             High Training Acc
   Low Validation Acc    →       High Validation Acc
   (Overfitting)                 (Good Generalization)
```

**Bias-Variance Tradeoff:**
```
Error
  ↑
  │                    Total Error
  │                   ╱
  │                  ╱
  │                 ╱
  │                ╱
  │               ╱
  │    Variance ╱
  │            ╱
  │           ╱
  │          ╱
  │         ╱
  │        ╱
  │       ╱
  │      ╱
  │     ╱
  │    ╱
  │   ╱
  │  ╱
  │ ╱
  │╱───────────────────────
  │    Bias
  │
  └────────────────────────→ Model Complexity
  
Optimal: Balance bias and variance
```

---

## 📚 L1 and L2 Regularization

### 8.3.1 L2 (Ridge) Regularization

**Core Idea:** Penalize large weights by adding squared magnitude to loss

**Loss Function:**
$$L_{total} = L_{original} + \lambda \sum_{i} w_i^2$$

Where:
- $L_{original}$: Original loss (MSE, Cross-Entropy, etc.)
- $\lambda$: Regularization strength
- $w_i$: Individual weights

**Gradient with L2:**
$$\frac{\partial L_{total}}{\partial w} = \frac{\partial L_{original}}{\partial w} + 2\lambda w$$

**Effect:**
- Weights decay toward zero
- Prevents any single weight from dominating
- Also called "weight decay"

**Visual Effect:**
```
Loss Landscape:
Without L2:          With L2:
   ___                  ___
  /   \                /   \
 /     \              /  ●  \   ← Minimum shifted
/       \            /       \      toward origin
●                    ●
Minimum              Minimum (regularized)
```

### 8.3.2 L1 (Lasso) Regularization

**Core Idea:** Add absolute value of weights to loss

**Loss Function:**
$$L_{total} = L_{original} + \lambda \sum_{i} |w_i|$$

**Gradient with L1:**
$$\frac{\partial L_{total}}{\partial w} = \frac{\partial L_{original}}{\partial w} + \lambda \cdot \text{sign}(w)$$

**Effect:**
- Drives weights to exactly zero
- Creates sparse models
- Feature selection effect

### 8.3.3 Elastic Net

**Combination of L1 and L2:**
$$L_{total} = L_{original} + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2$$

**Benefits:**
- Sparsity from L1
- Stability from L2
- Best of both worlds

### L1 vs L2 Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| **Penalty** | \|w\| | w² |
| **Sparsity** | ✅ Produces sparse weights | ❌ Small but non-zero weights |
| **Feature Selection** | ✅ Automatic | ❌ No |
| **Analytical Solution** | ❌ No | ✅ Yes |
| **Gradient** | Constant (sign) | Proportional to w |
| **Use Case** | Feature selection | General regularization |

### Implementation from Scratch

```python
import numpy as np

class RegularizedLinearRegression:
    """
    Linear regression with L1, L2, or Elastic Net regularization.
    """
    
    def __init__(self, 
                 regularization: str = 'l2',
                 lambda_l1: float = 0.0,
                 lambda_l2: float = 0.0,
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000):
        """
        Initialize Regularized Linear Regression.
        
        Args:
            regularization: Type ('l1', 'l2', 'elastic_net', None)
            lambda_l1: L1 regularization strength
            lambda_l2: L2 regularization strength
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of gradient descent iterations
        """
        self.regularization = regularization
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _compute_loss(self, X, y, predict=False):
        """Compute loss with regularization"""
        n_samples = len(y)
        
        if predict:
            y_pred = self.predict(X)
        else:
            y_pred = X @ self.weights + self.bias
        
        # Base loss (MSE)
        mse_loss = np.mean((y - y_pred) ** 2) / 2
        
        # Regularization loss
        if self.regularization == 'l1':
            reg_loss = self.lambda_l1 * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            reg_loss = self.lambda_l2 * np.sum(self.weights ** 2)
        elif self.regularization == 'elastic_net':
            reg_loss = (self.lambda_l1 * np.sum(np.abs(self.weights)) + 
                       self.lambda_l2 * np.sum(self.weights ** 2))
        else:
            reg_loss = 0
        
        return mse_loss + reg_loss
    
    def _compute_gradient(self, X, y):
        """Compute gradients with regularization"""
        n_samples = len(y)
        y_pred = X @ self.weights + self.bias
        
        # Base gradients
        dw = (1 / n_samples) * X.T @ (y_pred - y)
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        # Regularization gradients
        if self.regularization == 'l1':
            dw += self.lambda_l1 * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.lambda_l2 * self.weights
        elif self.regularization == 'elastic_net':
            dw += self.lambda_l1 * np.sign(self.weights)
            dw += 2 * self.lambda_l2 * self.weights
        
        return dw, db
    
    def fit(self, X, y):
        """Fit model with gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Compute loss
            loss = self._compute_loss(X)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw, db = self._compute_gradient(X, y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        """Predict target values"""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Neural Network Regularization
class RegularizedNeuralNetwork:
    """
    Neural network with L2 regularization (weight decay).
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, lambda_l2=0.01):
        """
        Initialize neural network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            learning_rate: Learning rate
            lambda_l2: L2 regularization strength
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.parameters = {}
        
        # Initialize weights (He initialization)
        for i in range(1, len(layer_sizes)):
            self.parameters[f'W{i}'] = np.random.randn(
                layer_sizes[i], layer_sizes[i-1]
            ) * np.sqrt(2.0 / layer_sizes[i-1])
            self.parameters[f'b{i}'] = np.zeros((layer_sizes[i], 1))
    
    def relu(self, Z):
        """ReLU activation"""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """ReLU derivative"""
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    
    def forward_propagation(self, X):
        """Forward propagation with caching"""
        caches = []
        A = X
        
        L = len(self.parameters) // 2
        
        for l in range(1, L):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = W @ A_prev + b
            A = self.relu(Z)
            
            caches.append((A_prev, Z))
        
        # Output layer
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        Z = W @ A + b
        AL = self.sigmoid(Z)
        caches.append((A, Z))
        
        return AL, caches
    
    def compute_cost(self, AL, Y):
        """Compute cross-entropy loss with L2 regularization"""
        m = Y.shape[1]
        L = len(self.parameters) // 2
        
        # Base cross-entropy loss
        cost = -np.mean(Y * np.log(AL + 1e-15) + (1 - Y) * np.log(1 - AL + 1e-15))
        
        # L2 regularization term
        l2_cost = 0
        for l in range(1, L + 1):
            l2_cost += np.sum(self.parameters[f'W{l}'] ** 2)
        l2_cost *= (self.lambda_l2 / (2 * m))
        
        return cost + l2_cost
    
    def backward_propagation(self, AL, Y, caches):
        """Backward propagation with L2 regularization"""
        m = AL.shape[1]
        L = len(self.parameters) // 2
        grads = {}
        
        # Output layer
        dAL = -(Y / (AL + 1e-15) - (1 - Y) / (1 - AL + 1e-15))
        A_prev, Z = caches[-1]
        dZ = dAL * self.sigmoid(Z) * (1 - self.sigmoid(Z))
        
        grads[f'dW{L}'] = (1 / m) * dZ @ A_prev.T + (self.lambda_l2 / m) * self.parameters[f'W{L}']
        grads[f'db{L}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers
        for l in reversed(range(1, L)):
            A_prev, Z = caches[l - 1]
            dA_prev = self.parameters[f'W{l+1}'].T @ dZ
            dZ = dA_prev * self.relu_derivative(Z)
            
            grads[f'dW{l}'] = (1 / m) * dZ @ A_prev.T + (self.lambda_l2 / m) * self.parameters[f'W{l}']
            grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        return grads
    
    def fit(self, X, Y, epochs=1000, print_cost=True):
        """Train the neural network"""
        costs = []
        
        for epoch in range(epochs):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(AL, Y)
            costs.append(cost)
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters
            for l in range(1, len(self.parameters) // 2 + 1):
                self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
                self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
            
            if print_cost and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
        
        return self, costs
    
    def predict(self, X):
        """Predict class labels"""
        AL, _ = self.forward_propagation(X)
        return (AL > 0.5).astype(int)
```

---

## 📚 Dropout

### 8.3.4 Dropout Algorithm

**Core Idea:** Randomly drop neurons during training to prevent co-adaptation

**Mechanism:**
```
Training:
    Input → [●●●●] → [●●●●] → Output
              50% dropout
    
    Iteration 1: [●0●0] → [●●0●]  (0 = dropped)
    Iteration 2: [0●0●] → [●0●0]
    Iteration 3: [●●00] → [0●●0]

Testing:
    Input → [●●●●] → [●●●●] → Output
              (all neurons, scaled weights)
```

**Mathematical Formulation:**
```
During training:
r ~ Bernoulli(p)  # Mask with probability p
h' = h ⊙ r        # Apply mask
output = f(h')

During testing:
output = f(p · h)  # Scale by p
```

**Inverted Dropout (more common):**
```
During training:
r ~ Bernoulli(p)
h' = (h ⊙ r) / p  # Scale during training
output = f(h')

During testing:
output = f(h)  # No scaling needed
```

### Dropout Implementation

```python
class DropoutLayer:
    """
    Dropout layer implementation.
    """
    
    def __init__(self, dropout_rate=0.5):
        """
        Initialize Dropout.
        
        Args:
            dropout_rate: Probability of dropping a neuron (0 to 1)
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, X):
        """Forward pass with dropout"""
        if self.training:
            # Generate mask
            self.mask = (np.random.rand(*X.shape) > self.dropout_rate).astype(float)
            # Apply mask and scale (inverted dropout)
            return X * self.mask / (1 - self.dropout_rate)
        else:
            # No dropout during inference
            return X
    
    def backward(self, dout):
        """Backward pass through dropout"""
        # Gradient flows only through non-dropped neurons
        return dout * self.mask / (1 - self.dropout_rate)
    
    def set_training(self, mode=True):
        """Set training mode"""
        self.training = mode


class NeuralNetworkWithDropout:
    """
    Neural network with dropout regularization.
    """
    
    def __init__(self, layer_sizes, dropout_rate=0.5, learning_rate=0.01):
        """
        Initialize network with dropout.
        
        Args:
            layer_sizes: List of layer sizes
            dropout_rate: Dropout probability
            learning_rate: Learning rate
        """
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.parameters = {}
        self.dropout_layers = {}
        
        # Initialize weights
        for i in range(1, len(layer_sizes)):
            self.parameters[f'W{i}'] = np.random.randn(
                layer_sizes[i], layer_sizes[i-1]
            ) * np.sqrt(2.0 / layer_sizes[i-1])
            self.parameters[f'b{i}'] = np.zeros((layer_sizes[i], 1))
            
            # Add dropout after each hidden layer
            if i < len(layer_sizes) - 1:
                self.dropout_layers[f'dropout{i}'] = DropoutLayer(dropout_rate)
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    
    def forward_propagation(self, X, training=True):
        """Forward propagation with dropout"""
        caches = []
        A = X
        
        L = len(self.parameters) // 2
        
        for l in range(1, L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = W @ A_prev + b
            
            if l < L:
                # Hidden layer: ReLU + Dropout
                A = self.relu(Z)
                if f'dropout{l}' in self.dropout_layers:
                    self.dropout_layers[f'dropout{l}'].set_training(training)
                    A = self.dropout_layers[f'dropout{l}'].forward(A)
            else:
                # Output layer: Sigmoid (no dropout)
                A = self.sigmoid(Z)
            
            caches.append((A_prev, Z))
        
        return A, caches
    
    def compute_cost(self, AL, Y):
        """Compute cross-entropy loss"""
        m = Y.shape[1]
        cost = -np.mean(Y * np.log(AL + 1e-15) + (1 - Y) * np.log(1 - AL + 1e-15))
        return cost
    
    def backward_propagation(self, AL, Y, caches):
        """Backward propagation with dropout"""
        m = AL.shape[1]
        L = len(self.parameters) // 2
        grads = {}
        
        # Output layer
        dAL = -(Y / (AL + 1e-15) - (1 - Y) / (1 - AL + 1e-15))
        A_prev, Z = caches[-1]
        dZ = dAL * self.sigmoid(Z) * (1 - self.sigmoid(Z))
        
        grads[f'dW{L}'] = (1 / m) * dZ @ A_prev.T
        grads[f'db{L}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers (with dropout)
        for l in reversed(range(1, L)):
            # Apply dropout gradient
            if f'dropout{l+1}' in self.dropout_layers:
                dA_prev = self.parameters[f'W{l+1}'].T @ dZ
                dA_prev = self.dropout_layers[f'dropout{l+1}'].backward(dA_prev)
            else:
                dA_prev = self.parameters[f'W{l+1}'].T @ dZ
            
            A_prev, Z = caches[l - 1] if l > 0 else (None, None)
            if l > 0:
                dZ = dA_prev * self.relu_derivative(Z)
                
                grads[f'dW{l}'] = (1 / m) * dZ @ A_prev.T
                grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        return grads
    
    def fit(self, X, Y, epochs=1000, print_cost=True):
        """Train with dropout"""
        costs = []
        
        for epoch in range(epochs):
            # Forward (training mode)
            AL, caches = self.forward_propagation(X, training=True)
            
            # Compute cost
            cost = self.compute_cost(AL, Y)
            costs.append(cost)
            
            # Backward
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update
            for l in range(1, L + 1):
                self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
                self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
            
            if print_cost and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
        
        return self, costs
    
    def predict(self, X):
        """Predict (inference mode)"""
        AL, _ = self.forward_propagation(X, training=False)
        return (AL > 0.5).astype(int)


# Dropout Variants
class SpatialDropout:
    """
    Spatial Dropout for CNNs - drops entire feature maps.
    """
    
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, X, training=True):
        """
        X shape: (batch, channels, height, width)
        """
        if training:
            # Create mask per sample, per channel
            batch_size, channels, _, _ = X.shape
            self.mask = (np.random.rand(batch_size, channels, 1, 1) > self.dropout_rate).astype(float)
            return X * self.mask / (1 - self.dropout_rate)
        return X


class DropConnect:
    """
    DropConnect - drops weights instead of activations.
    """
    
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
    
    def forward(self, W, training=True):
        """Apply DropConnect to weights"""
        if training:
            self.mask = (np.random.rand(*W.shape) > self.drop_rate).astype(float)
            return W * self.mask / (1 - self.drop_rate)
        return W
```

### Dropout Best Practices

| Aspect | Recommendation |
|--------|----------------|
| **Dropout Rate** | 0.2-0.5 for hidden layers |
| **Input Layer** | Usually no dropout or very low (0.1) |
| **Output Layer** | No dropout |
| **Small Networks** | Lower dropout (0.2-0.3) |
| **Large Networks** | Higher dropout (0.5-0.7) |
| **CNNs** | Use Spatial Dropout or lower rates |
| **RNNs** | Use variational dropout (same mask across time) |

---

## 📚 Batch Normalization

### 8.3.5 BatchNorm Fundamentals

**Problem:** Internal Covariate Shift

**Definition:** Change in distribution of layer inputs during training

**Consequences:**
- Requires lower learning rates
- Careful initialization needed
- Training slows down

**Solution:** Normalize layer inputs

### BatchNorm Algorithm

**Forward Pass (Training):**
```
1. Compute batch mean: μ_B = (1/m) Σ x_i
2. Compute batch variance: σ²_B = (1/m) Σ (x_i - μ_B)²
3. Normalize: x̂_i = (x_i - μ_B) / √(σ²_B + ε)
4. Scale and shift: y_i = γ · x̂_i + β
```

**Forward Pass (Inference):**
```
Use running averages of μ and σ²:
y = γ · (x - μ_running) / √(σ²_running + ε) + β
```

**Visual:**
```
Before BatchNorm:          After BatchNorm:
Layer 1 output             Layer 1 output
   ↓ (changing dist)          ↓ (normalized)
Layer 2 input              BatchNorm
   ↓                          ↓
                        Layer 2 input
                        (stable dist)
```

### BatchNorm Implementation

```python
class BatchNormalization:
    """
    Batch Normalization layer.
    """
    
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        """
        Initialize BatchNorm.
        
        Args:
            num_features: Number of features (C in NxCxHxW or NxC)
            epsilon: Small constant for numerical stability
            momentum: Momentum for running averages
        """
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale
        self.beta = np.zeros(num_features)  # Shift
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward pass
        self.cache = None
        self.training = True
    
    def forward(self, X):
        """
        Forward pass.
        
        Args:
            X: Input of shape (N, C) or (N, C, H, W)
        """
        if len(X.shape) == 2:
            # Fully connected: (N, C)
            return self._forward_fc(X)
        elif len(X.shape) == 4:
            # Convolutional: (N, C, H, W)
            return self._forward_conv(X)
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")
    
    def _forward_fc(self, X):
        """BatchNorm for fully connected layers"""
        if self.training:
            # Batch statistics
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)
            
            # Normalize
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Scale and shift
            out = self.gamma * X_norm + self.beta
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Cache for backward
            self.cache = (X, X_norm, batch_mean, batch_var)
        else:
            # Use running statistics
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * X_norm + self.beta
        
        return out
    
    def _forward_conv(self, X):
        """BatchNorm for convolutional layers"""
        N, C, H, W = X.shape
        
        if self.training:
            # Reshape for per-channel statistics
            X_reshaped = X.transpose(0, 2, 3, 1).reshape(-1, C)
            
            batch_mean = np.mean(X_reshaped, axis=0)
            batch_var = np.var(X_reshaped, axis=0)
            
            X_norm = (X_reshaped - batch_mean) / np.sqrt(batch_var + self.epsilon)
            X_norm = X_norm.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
            out = self.gamma.reshape(1, C, 1, 1) * X_norm + self.beta.reshape(1, C, 1, 1)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            self.cache = (X, X_norm, batch_mean, batch_var)
        else:
            X_reshaped = X.transpose(0, 2, 3, 1).reshape(-1, C)
            X_norm = (X_reshaped - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            X_norm = X_norm.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
            out = self.gamma.reshape(1, C, 1, 1) * X_norm + self.beta.reshape(1, C, 1, 1)
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Args:
            dout: Upstream gradient
        """
        if len(dout.shape) == 2:
            return self._backward_fc(dout)
        elif len(dout.shape) == 4:
            return self._backward_conv(dout)
    
    def _backward_fc(self, dout):
        """Backward for fully connected"""
        X, X_norm, mean, var = self.cache
        N = X.shape[0]
        
        # Gradients for gamma and beta
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        # Gradient for X_norm
        dX_norm = dout * self.gamma
        
        # Gradient for variance
        dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + self.epsilon) ** (-1.5), axis=0)
        
        # Gradient for mean
        dmean = np.sum(dX_norm * -1 / np.sqrt(var + self.epsilon), axis=0)
        dmean += dvar * np.sum(-2 * (X - mean), axis=0) / N
        
        # Gradient for X
        dX = dX_norm / np.sqrt(var + self.epsilon)
        dX += dvar * 2 * (X - mean) / N
        dX += dmean / N
        
        return dX, dgamma, dbeta
    
    def _backward_conv(self, dout):
        """Backward for convolutional"""
        N, C, H, W = dout.shape
        X, X_norm, mean, var = self.cache
        
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        X_reshaped = X.transpose(0, 2, 3, 1).reshape(-1, C)
        X_norm_reshaped = X_norm.transpose(0, 2, 3, 1).reshape(-1, C)
        
        dgamma = np.sum(dout_reshaped * X_norm_reshaped, axis=0)
        dbeta = np.sum(dout_reshaped, axis=0)
        
        dX_norm = (dout * self.gamma.reshape(1, C, 1, 1))
        dX_norm = dX_norm.transpose(0, 2, 3, 1).reshape(-1, C)
        
        dvar = np.sum(dX_norm * (X_reshaped - mean) * -0.5 * (var + self.epsilon) ** (-1.5), axis=0)
        dmean = np.sum(dX_norm * -1 / np.sqrt(var + self.epsilon), axis=0)
        
        dX = dX_norm / np.sqrt(var + self.epsilon)
        dX += dvar * 2 * (X_reshaped - mean) / (N * H * W)
        dX += dmean / (N * H * W)
        
        dX = dX.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        
        return dX, dgamma, dbeta
    
    def set_training(self, mode=True):
        """Set training mode"""
        self.training = mode
```

### Benefits of BatchNorm

1. **Allows higher learning rates**
2. **Reduces sensitivity to initialization**
3. **Acts as regularizer** (reduces need for dropout)
4. **Accelerates training**
5. **Slightly improves accuracy**

---

## 📚 Other Normalization Techniques

### 8.3.6 Layer Normalization

**Problem with BatchNorm:** 
- Doesn't work well with small batches
- Not suitable for RNNs (sequence length varies)

**Solution:** Normalize across features instead of batch

**LayerNorm:**
$$\mu_i = \frac{1}{H} \sum_{j=1}^{H} x_{ij}$$
$$\sigma_i^2 = \frac{1}{H} \sum_{j=1}^{H} (x_{ij} - \mu_i)^2$$
$$\hat{x}_{ij} = \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$
$$y_{ij} = \gamma_j \hat{x}_{ij} + \beta_j$$

```python
class LayerNormalization:
    """Layer Normalization"""
    
    def __init__(self, num_features, epsilon=1e-5):
        self.num_features = num_features
        self.epsilon = epsilon
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
    
    def forward(self, X):
        """
        X: (N, D) or (N, T, D) for sequences
        """
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        
        X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        
        return out
```

### Instance Normalization

**Use Case:** Style transfer, image generation

**InstanceNorm:**
- Normalize each sample independently
- No batch dependency

```python
class InstanceNormalization:
    """Instance Normalization for style transfer"""
    
    def __init__(self, num_features, epsilon=1e-5):
        self.num_features = num_features
        self.epsilon = epsilon
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
    
    def forward(self, X):
        """
        X: (N, C, H, W)
        """
        mean = np.mean(X, axis=(2, 3), keepdims=True)
        var = np.var(X, axis=(2, 3), keepdims=True)
        
        X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        
        return out
```

### Group Normalization

**Idea:** Divide channels into groups, normalize within each group

```python
class GroupNormalization:
    """
    Group Normalization - works with any batch size.
    """
    
    def __init__(self, num_features, num_groups=32, epsilon=1e-5):
        self.num_features = num_features
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
    
    def forward(self, X):
        """
        X: (N, C, H, W)
        """
        N, C, H, W = X.shape
        
        # Reshape to (N, G, C/G, H, W)
        X_reshaped = X.reshape(N, self.num_groups, C // self.num_groups, H, W)
        
        mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
        var = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
        
        X_norm = (X_reshaped - mean) / np.sqrt(var + self.epsilon)
        X_norm = X_norm.reshape(N, C, H, W)
        
        out = self.gamma * X_norm + self.beta
        
        return out
```

### Normalization Comparison

| Method | Normalizes Over | Batch Size Dependent | Best For |
|--------|-----------------|---------------------|----------|
| **BatchNorm** | Batch dimension | ✅ Yes | CNNs, large batches |
| **LayerNorm** | Feature dimension | ❌ No | RNNs, Transformers |
| **InstanceNorm** | Spatial (per sample) | ❌ No | Style transfer |
| **GroupNorm** | Groups of channels | ❌ No | Small batch CNNs |

---

## 📚 Other Regularization Methods

### 8.3.7 Early Stopping

**Idea:** Stop training when validation performance stops improving

```python
class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self, patience=10, min_delta=0.0001, restore_best=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, val_loss, model=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best and model:
                self.best_weights = model.get_weights()
        
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and model and self.best_weights:
                    model.set_weights(self.best_weights)
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best and model:
                self.best_weights = model.get_weights()
        
        return self.should_stop
```

### Data Augmentation

**Idea:** Artificially increase training data size

```python
class DataAugmentation:
    """Data augmentation for images"""
    
    def __init__(self, rotation_range=20, width_shift=0.2, 
                 height_shift=0.2, flip_horizontal=True):
        self.rotation_range = rotation_range
        self.width_shift = width_shift
        self.height_shift = height_shift
        self.flip_horizontal = flip_horizontal
    
    def augment(self, image):
        """Apply random augmentations"""
        # Random rotation
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        image = self._rotate(image, angle)
        
        # Random shift
        h_shift = np.random.uniform(-self.height_shift, self.height_shift)
        w_shift = np.random.uniform(-self.width_shift, self.width_shift)
        image = self._shift(image, h_shift, w_shift)
        
        # Random flip
        if self.flip_horizontal and np.random.random() > 0.5:
            image = np.fliplr(image)
        
        return image
    
    def _rotate(self, image, angle):
        """Rotate image"""
        from scipy import ndimage
        return ndimage.rotate(image, angle, reshape=False)
    
    def _shift(self, image, h_shift, w_shift):
        """Shift image"""
        from scipy import ndimage
        h, w = image.shape[:2]
        shift = (int(h * h_shift), int(w * w_shift))
        return ndimage.shift(image, shift)
```

### Label Smoothing

**Problem:** Model becomes overconfident

**Solution:** Soften target labels

```python
def label_smoothing(labels, epsilon=0.1):
    """
    Apply label smoothing.
    
    Args:
        labels: One-hot encoded labels
        epsilon: Smoothing factor
    
    Returns:
        Smoothed labels
    """
    n_classes = labels.shape[1]
    return (1 - epsilon) * labels + epsilon / n_classes


# Example usage
# Original: [0, 1, 0]
# Smoothed: [0.033, 0.933, 0.033]  (with ε=0.1)
```

---

## 📊 Summary Tables

### Regularization Techniques Overview

| Technique | Mechanism | Best For | Hyperparameters |
|-----------|-----------|----------|-----------------|
| **L1** | Sparse weights | Feature selection | λ (0.0001-0.1) |
| **L2** | Weight decay | General regularization | λ (0.0001-0.1) |
| **Dropout** | Random neuron dropping | Fully connected layers | rate (0.2-0.5) |
| **BatchNorm** | Normalize activations | CNNs | momentum (0.1) |
| **LayerNorm** | Normalize features | RNNs, Transformers | - |
| **Early Stopping** | Stop at validation peak | All models | patience (10-50) |
| **Data Augmentation** | Increase training data | Computer Vision | varies |
| **Label Smoothing** | Soften targets | Classification | ε (0.1) |

### When to Use Each Technique

| Scenario | Recommended Techniques |
|----------|----------------------|
| **Overfitting (FC layers)** | Dropout + L2 |
| **Overfitting (CNN)** | Data augmentation + L2 |
| **Slow training** | BatchNorm |
| **Small batch size** | GroupNorm or LayerNorm |
| **RNN/Transformer** | LayerNorm + Dropout |
| **Feature selection needed** | L1 regularization |
| **Model too confident** | Label smoothing |

---

## 🎯 ML Applications

| Application | Techniques Used | Purpose |
|-------------|-----------------|---------|
| **Image Classification** | BatchNorm, Dropout, Data Aug | Prevent overfitting |
| **Object Detection** | BatchNorm, L2, Focal Loss | Handle class imbalance |
| **NLP/Transformers** | LayerNorm, Dropout | Stabilize training |
| **GANs** | InstanceNorm, Spectral Norm | Improve stability |
| **Style Transfer** | InstanceNorm | Remove instance info |
| **Medical Imaging** | Heavy augmentation | Limited data |

---

## 📝 Practice Problems

### Level 1: Basic

1. **Conceptual**: Explain the bias-variance tradeoff
2. **Calculation**: Calculate L1 and L2 penalty for given weights
3. **Understanding**: Why does dropout work better during training than testing?
4. **Code**: Implement L2 regularization for a simple linear model
5. **Analysis**: Compare BatchNorm and LayerNorm normalization axes

### Level 2: Intermediate

1. **Implementation**: Build a neural network with dropout from scratch
2. **Experiment**: Compare training with and without BatchNorm
3. **Analysis**: Investigate effect of different dropout rates on accuracy
4. **Application**: Implement early stopping for a classification task
5. **Visualization**: Plot training curves with different regularization

### Level 3: Advanced

1. **Research**: Implement and compare all normalization techniques
2. **Optimization**: Combine multiple regularization techniques effectively
3. **Extension**: Implement variational dropout for RNNs
4. **Project**: Build image classifier with heavy regularization
5. **Analysis**: Study interaction between BatchNorm and dropout

---

## 🔗 Related Topics
- [[01-Neural-Network-Basics]] - Foundation for regularization
- [[02-Optimization-Algorithms]] - Optimizers with regularization
- [[04-Training-Deep-Networks]] - Debugging regularized networks
- [[05-Convolutional-Neural-Networks]] - CNN-specific regularization

---

**Status:** ✅ Complete  
**Next:** [[04-Training-Deep-Networks]]
