# Deep Learning Fundamentals - Practice Problems

## Topic 1: Neural Network Basics

### Level 1: Basic

**1.1** Implement a perceptron from scratch:
```python
class Perceptron:
    def __init__(self, input_size, lr=0.01):
        # Initialize weights and bias
        pass
    
    def sigmoid(self, z):
        # Sigmoid activation
        pass
    
    def predict(self, X):
        # Forward pass
        pass
    
    def fit(self, X, y, epochs=100):
        # Training loop
        pass

# Test on AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
```

**1.2** Compare activation functions:
```python
# Plot sigmoid, tanh, ReLU, Leaky ReLU, GELU
# Calculate and plot their derivatives
# Discuss vanishing gradient problem
```

### Level 2: Intermediate

**1.3** Build MLP from scratch:
```python
class MLP:
    def __init__(self, layer_sizes):
        # Initialize weights with He initialization
        pass
    
    def forward(self, X):
        # Forward propagation
        pass
    
    def backward(self, X, y, lr=0.01):
        # Backpropagation
        pass
    
    def fit(self, X, y, epochs=1000):
        # Training loop
        pass

# Test on XOR problem
```

---

## Topic 2: Optimization Algorithms

### Level 2: Intermediate

**2.1** Implement optimizers:
```python
# Implement from scratch:
# 1. SGD with Momentum
# 2. RMSprop
# 3. Adam
# 4. AdamW

# Compare convergence on same problem
```

**2.2** Learning rate scheduling:
```python
def cosine_annealing(epoch, total_epochs, min_lr=0.001, max_lr=0.01):
    # Implement cosine annealing
    pass

def warm_restarts(epoch, restart_period=10):
    # Implement warm restarts
    pass

# Plot learning rate schedules
```

### Level 3: Advanced

**2.3** Compare optimizer performance:
```python
# Train same network with different optimizers
# Plot loss curves
# Compare final accuracy
# Analyze convergence speed
```

---

## Topic 3: Regularization

### Level 1: Basic

**3.1** Implement regularization techniques:
```python
# 1. L1 regularization
# 2. L2 regularization
# 3. Elastic Net

# Compare effect on weights
```

**3.2** Implement Dropout:
```python
class DropoutLayer:
    def __init__(self, rate=0.5):
        pass
    
    def forward(self, X, training=True):
        pass
    
    def backward(self, dout):
        pass
```

### Level 2: Intermediate

**3.3** Implement Batch Normalization:
```python
class BatchNorm:
    def __init__(self, num_features, momentum=0.1):
        pass
    
    def forward(self, X, training=True):
        pass
    
    def backward(self, dout):
        pass
```

---

## Topic 4: CNNs

### Level 2: Intermediate

**4.1** Implement convolution from scratch:
```python
def conv2d(image, kernel, stride=1, padding='valid'):
    # 2D convolution implementation
    pass

# Test with different kernels:
# - Edge detection
# - Sharpening
# - Blurring
```

**4.2** Build CNN for MNIST:
```python
class SimpleCNN(keras.Model):
    def __init__(self):
        super().__init__()
        # Conv -> Pool -> Conv -> Pool -> FC
        pass
    
    def call(self, inputs):
        pass

# Train and achieve >98% accuracy
```

### Level 3: Advanced

**4.3** Implement pooling layers:
```python
def max_pool(X, pool_size=2, stride=2):
    pass

def average_pool(X, pool_size=2, stride=2):
    pass
```

---

## Topic 5: RNNs

### Level 3: Advanced

**5.1** Implement Vanilla RNN:
```python
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        pass
    
    def forward(self, X, h_prev=None):
        pass
```

**5.2** Implement LSTM:
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        # Forget gate, Input gate, Cell state, Output gate
        pass
    
    def forward(self, X, h_prev=None, c_prev=None):
        pass
```

**5.3** Text generation:
```python
# Train character-level RNN on Shakespeare text
# Generate new text in Shakespeare style
```

---

## Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1 Perceptron
```python
class Perceptron:
    def __init__(self, input_size, lr=0.01):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = lr
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(z) > 0.5).astype(int)
    
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            errors = y - predictions
            
            self.weights += self.lr * np.dot(X.T, errors)
            self.bias += self.lr * np.sum(errors)
```

### 2.1 Adam Optimizer
```python
def adam(X, y, lr=0.001, epochs=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    m_dw, v_dw = np.zeros(n), np.zeros(n)
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        dw = (2/m) * np.dot(X.T, (predictions - y))
        db = (2/m) * np.sum(predictions - y)
        
        m_dw = beta1 * m_dw + (1 - beta1) * dw
        v_dw = beta2 * v_dw + (1 - beta2) * dw**2
        
        m_dw_hat = m_dw / (1 - beta1**(epoch+1))
        v_dw_hat = v_dw / (1 - beta2**(epoch+1))
        
        weights -= lr * m_dw_hat / (np.sqrt(v_dw_hat) + epsilon)
        bias -= lr * db / (1 - beta1**(epoch+1))
    
    return weights, bias
```

### 3.2 Dropout
```python
class DropoutLayer:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
    
    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.rate).astype(float)
            self.mask /= (1 - self.rate)  # Inverted dropout
            return X * self.mask
        else:
            return X
    
    def backward(self, dout):
        return dout * self.mask
```

### 4.1 Convolution
```python
def conv2d(image, kernel, stride=1, padding='valid'):
    if padding == 'same':
        pad = kernel.shape[0] // 2
        image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
    
    ih, iw = image.shape
    kh, kw = kernel.shape
    
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1
    
    output = np.zeros((oh, ow))
    
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel)
    
    return output
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Deep Learning Fundamentals Complete!
