# 8.1 Neural Network Basics

## 🎯 Quick Overview
- **Perceptron**: Basic unit of neural networks
- **MLP**: Multi-layer architectures
- **Activation Functions**: Non-linearity
- **Loss Functions**: Measure prediction error
- **Backpropagation**: Training algorithm
- **Foundation for**: All deep learning

---

## 1. Biological Inspiration

### Biological vs Artificial Neurons

```
Biological Neuron:
- Dendrites (input)
- Cell body (processing)
- Axon (output)
- Synapses (connections)

Artificial Neuron:
- Inputs (x₁, x₂, ..., xₙ)
- Weights (w₁, w₂, ..., wₙ)
- Bias (b)
- Activation function (σ)
- Output (y)
```

### Perceptron Model

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = y - predictions
            
            # Update weights and bias
            self.weights += self.lr * np.dot(X.T, errors)
            self.bias += self.lr * np.sum(errors)
            
            if epoch % 10 == 0:
                loss = np.mean(errors ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])  # OR gate

perceptron = Perceptron(input_size=2, learning_rate=0.1)
perceptron.fit(X, y, epochs=100)
```

### Perceptron Limitations (XOR Problem)

```python
# XOR problem - not linearly separable
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Single perceptron cannot solve XOR
# Need multi-layer network (MLP)
```

---

## 2. Multi-Layer Perceptron (MLP)

### Architecture

```python
class MLP:
    def __init__(self, layer_sizes):
        """
        layer_sizes: [input_size, hidden1_size, hidden2_size, ..., output_size]
        """
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(current, w) + b
            self.z_values.append(z)
            
            # Apply ReLU for all layers except last
            if len(self.z_values) < len(self.weights):
                current = self.relu(z)
            else:
                current = z  # Output layer (no activation for regression)
        
        self.activations.append(current)
        return current
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        output = self.activations[-1]
        
        # Output layer error
        delta = output - y.reshape(-1, 1)
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean((self.activations[-1] - y.reshape(-1, 1)) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return self.forward(X)

# Example: XOR with MLP
mlp = MLP(layer_sizes=[2, 4, 1])
mlp.fit(X_xor, y_xor, epochs=1000, learning_rate=0.1)
predictions = mlp.predict(X_xor)
print(f"Predictions: {predictions.flatten()}")
```

---

## 3. Activation Functions

### Comparison of Activation Functions

```python
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def swish(x):
    return x * sigmoid(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Plot activation functions
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 2)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 3)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 5)
plt.plot(x, elu(x))
plt.title('ELU')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 6)
plt.plot(x, gelu(x))
plt.title('GELU')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 7)
plt.plot(x, swish(x))
plt.title('Swish')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Activation Function Selection

```python
# Guidelines:
# - Hidden layers: ReLU, Leaky ReLU, GELU (default: ReLU)
# - Binary classification output: Sigmoid
# - Multi-class classification output: Softmax
# - Regression output: Linear (no activation)
```

---

## 4. Loss Functions

### Regression Losses

```python
def mse_loss(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss (robust to outliers)"""
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

def log_cosh_loss(y_true, y_pred):
    """Log-Cosh Loss"""
    error = y_true - y_pred
    return np.mean(np.log(np.cosh(error)))
```

### Classification Losses

```python
def binary_cross_entropy(y_true, y_pred):
    """Binary Cross-Entropy"""
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    """Categorical Cross-Entropy (one-hot encoded)"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred), axis=1).mean()

def sparse_categorical_cross_entropy(y_true, y_pred):
    """Sparse Categorical Cross-Entropy (integer labels)"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n_samples = y_true.shape[0]
    return -np.mean(np.log(y_pred[np.arange(n_samples), y_true]))

def hinge_loss(y_true, y_pred):
    """Hinge Loss (for SVM)"""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def focal_loss(y_true, y_pred, gamma=2.0):
    """Focal Loss (for imbalanced datasets)"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    return -np.mean((1 - pt) ** gamma * np.log(pt))
```

---

## 5. Backpropagation

### Computational Graph

```python
# Example: Simple computational graph
# z = (x + y) * w
# Loss = (z - target)²

def forward(x, y, w, target):
    # Forward pass
    a = x + y
    z = a * w
    loss = (z - target) ** 2
    
    # Backward pass (chain rule)
    dloss_dz = 2 * (z - target)
    dz_dw = a
    dz_da = w
    da_dx = 1
    da_dy = 1
    
    # Gradients
    dloss_dw = dloss_dz * dz_dw
    dloss_da = dloss_dz * dz_da
    dloss_dx = dloss_da * da_dx
    dloss_dy = dloss_da * da_dy
    
    return loss, {'dw': dloss_dw, 'dx': dloss_dx, 'dy': dloss_dy}

# Example
x, y, w = 2, 3, 4
target = 20
loss, grads = forward(x, y, w, target)
print(f"Loss: {loss}")
print(f"Gradients: {grads}")
```

### Vectorized Backpropagation

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(current, w) + b
            self.z_values.append(z)
            current = self.relu(z) if len(self.z_values) < len(self.weights) else z
        
        self.activations.append(current)
        return current
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        output = self.activations[-1]
        
        # Output layer gradient (MSE loss derivative)
        delta = 2 * (output - y.reshape(-1, 1)) / m
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients for current layer
            dw = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            losses.append(loss)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        return self.forward(X)

# Example usage
nn = NeuralNetwork([2, 4, 1])
losses = nn.fit(X_xor, y_xor, epochs=1000, learning_rate=0.1)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 💻 Python Code Examples

```python
# === Complete Neural Network from Scratch ===

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15, n_redundant=5,
                           random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train network
nn = NeuralNetwork([20, 32, 16, 1])
losses = nn.fit(X_train_scaled, y_train, epochs=2000, learning_rate=0.01)

# Evaluate
train_pred = nn.predict(X_train_scaled)
test_pred = nn.predict(X_test_scaled)

train_accuracy = np.mean((train_pred > 0.5).flatten() == y_train)
test_accuracy = np.mean((test_pred > 0.5).flatten() == y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(train_pred[y_train == 0], bins=20, alpha=0.5, label='Class 0')
plt.hist(train_pred[y_train == 1], bins=20, alpha=0.5, label='Class 1')
plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Prediction Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 📊 Summary Tables

### Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Binary output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Hidden layers |
| ReLU | max(0, x) | [0, ∞) | Default hidden |
| Leaky ReLU | max(αx, x) | (-∞, ∞) | Avoid dead ReLU |
| Softmax | eˣⁱ/Σeˣʲ | (0, 1) | Multi-class output |

### Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| MSE | (y-ŷ)² | Regression |
| MAE | \|y-ŷ\| | Regression (robust) |
| Binary CE | -[y·log(ŷ)+(1-y)·log(1-ŷ)] | Binary classification |
| Categorical CE | -Σy·log(ŷ) | Multi-class classification |
| Hinge | max(0, 1-y·ŷ) | SVM |

---

## 🎯 ML Applications

| Concept | ML Application |
|---------|----------------|
| Perceptron | Binary classification |
| MLP | General pattern recognition |
| ReLU | Deep neural networks |
| Softmax | Multi-class classification |
| Cross-Entropy | Classification tasks |
| Backpropagation | Neural network training |

---

**Status:** ✅ Complete
**Next:** Optimization Algorithms
