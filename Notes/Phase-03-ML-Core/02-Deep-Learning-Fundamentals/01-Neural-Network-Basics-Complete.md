# 8.1 Neural Network Basics - Complete Expanded Guide

## 🎯 Learning Objectives

After completing this section, you will master:
1. **Biological foundations** - How brain neurons inspired artificial neurons
2. **Perceptron implementation** - Build from scratch with full understanding
3. **Multi-layer networks** - Why depth matters and how to implement
4. **Activation functions** - When and why to use each type
5. **Loss functions** - How to measure and minimize errors
6. **Backpropagation** - The algorithm that makes deep learning possible
7. **Complete implementations** - Production-ready code from scratch

---

## Part 1: Biological Inspiration (8.1.1)

### The Human Brain

```
The human brain contains:
- ~86 billion neurons
- ~100 trillion synapses (connections)
- Each neuron connects to ~1000-10000 other neurons
- Signals travel at speeds up to 120 m/s

Key insight: Intelligence emerges from simple units working together!
```

### Biological Neuron Structure

```
┌─────────────────────────────────────────────────────────┐
│                  BIOLOGICAL NEURON                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│    Dendrites ← Receive signals from other neurons       │
│       ↓                                                  │
│    Cell Body (Soma) ← Integrates all signals           │
│       ↓                                                  │
│    Axon Hillock ← Decision point (fire or not?)        │
│       ↓                                                  │
│    Axon ← Transmits signal                              │
│       ↓                                                  │
│    Axon Terminals → Synapses → Other neurons           │
│                                                          │
└─────────────────────────────────────────────────────────┘

Signal Flow:
1. Dendrites receive chemical signals
2. Signals convert to electrical impulses
3. Cell body integrates all inputs
4. If threshold exceeded → action potential fires
5. Signal travels down axon
6. Neurotransmitters released at synapses
7. Process repeats in next neuron
```

### Hebbian Learning - The Basis of AI Learning

```
Donald Hebb (1949):
"Cells that fire together, wire together"

Meaning:
- When neuron A repeatedly helps fire neuron B
- The connection (synapse) between them strengthens
- This is how memories and learning occur in the brain!

Artificial Neural Network Equivalent:
- Weights represent synapse strengths
- Training adjusts weights based on error
- Stronger connections = larger weights
```

### From Biology to Mathematics

```
Biological Process          →    Mathematical Model
─────────────────────────────────────────────────────
Dendrite inputs             →    Input features (x₁, x₂, ...)
Synapse strength            →    Weights (w₁, w₂, ...)
Signal integration          →    Weighted sum (Σwᵢxᵢ)
Firing threshold            →    Bias (b)
Action potential            →    Activation function σ(z)
Axon output                 →    Output (y)
```

---

## Part 2: The Perceptron - Complete Implementation (8.1.2)

### Mathematical Foundation

```
The perceptron computes:

Step 1: Weighted Sum (Linear Combination)
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

In vector notation:
z = w·x + b

Where:
- x ∈ ℝⁿ: Input vector (n features)
- w ∈ ℝⁿ: Weight vector (n weights)
- b ∈ ℝ: Bias (scalar)
- z ∈ ℝ: Weighted sum (scalar)

Step 2: Activation (Non-linear Transformation)
y = σ(z)

For perceptron, σ is typically:
- Step function (original perceptron)
- Sigmoid (modern variant)
```

### Complete Perceptron Class

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class Perceptron:
    """
    Perceptron: The fundamental unit of neural networks
    
    This implementation includes:
    - Multiple activation functions
    - Comprehensive training metrics
    - Visualization tools
    - Production-ready code
    
    Parameters
    ----------
    input_size : int
        Number of input features
    learning_rate : float, default=0.01
        Step size for weight updates (0.001 to 0.1 typical)
    n_iterations : int, default=1000
        Number of training iterations
    activation : str, default='sigmoid'
        Activation function: 'sigmoid', 'tanh', 'relu', 'step'
    
    Attributes
    ----------
    weights : ndarray
        Weight vector of shape (input_size,)
    bias : float
        Bias term
    loss_history : list
        Loss value at each iteration
    """
    
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        activation: str = 'sigmoid'
    ):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_name = activation
        
        # Initialize weights using Xavier initialization
        # This helps with convergence
        self.weights = np.random.randn(input_size) * np.sqrt(2 / input_size)
        self.bias = 0.0
        
        # Training history
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        
        # Select activation function
        self._set_activation(activation)
    
    def _set_activation(self, activation: str):
        """Set activation function and its derivative"""
        if activation == 'sigmoid':
            self.activation = self._sigmoid
            self.activation_deriv = self._sigmoid_derivative
        elif activation == 'tanh':
            self.activation = self._tanh
            self.activation_deriv = self._tanh_derivative
        elif activation == 'relu':
            self.activation = self._relu
            self.activation_deriv = self._relu_derivative
        elif activation == 'step':
            self.activation = self._step
            self.activation_deriv = None  # Step has no useful derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    # ========== ACTIVATION FUNCTIONS ==========
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Properties:
        - Output range: (0, 1)
        - Zero-centered: No
        - Differentiable: Yes
        - Use case: Binary classification output
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid
        
        Formula: σ'(z) = σ(z) × (1 - σ(z))
        
        This is used in backpropagation!
        """
        sig = self._sigmoid(z)
        return sig * (1 - sig)
    
    def _tanh(self, z: np.ndarray) -> np.ndarray:
        """
        Hyperbolic tangent activation
        
        Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
        
        Properties:
        - Output range: (-1, 1)
        - Zero-centered: Yes
        - Use case: Hidden layers (better than sigmoid)
        """
        return np.tanh(z)
    
    def _tanh_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of tanh: 1 - tanh²(z)"""
        return 1 - np.tanh(z) ** 2
    
    def _relu(self, z: np.ndarray) -> np.ndarray:
        """
        Rectified Linear Unit (ReLU)
        
        Formula: ReLU(z) = max(0, z)
        
        Properties:
        - Output range: [0, ∞)
        - Zero-centered: No
        - Use case: Hidden layers (default choice)
        - Advantage: Solves vanishing gradient
        - Disadvantage: Dying ReLU problem
        """
        return np.maximum(0, z)
    
    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU
        
        Formula:
        ReLU'(z) = 1 if z > 0
                 = 0 if z ≤ 0
        """
        return (z > 0).astype(float)
    
    def _step(self, z: np.ndarray) -> np.ndarray:
        """
        Step function (original perceptron)
        
        Formula:
        step(z) = 1 if z ≥ 0
                = 0 if z < 0
        
        Note: No useful derivative, use only for inference
        """
        return (z >= 0).astype(int)
    
    # ========== FORWARD PASS ==========
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation
        
        Computes: y = σ(X·w + b)
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Output predictions
        """
        # Linear combination: z = X·w + b
        z = np.dot(X, self.weights) + self.bias
        
        # Apply activation: y = σ(z)
        predictions = self.activation(z)
        
        return predictions
    
    # ========== LOSS FUNCTIONS ==========
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute loss based on activation function
        
        For sigmoid/tanh: Binary Cross-Entropy
        For others: Mean Squared Error
        """
        if self.activation_name in ['sigmoid', 'tanh']:
            return self._binary_cross_entropy(y_true, y_pred)
        else:
            return self._mean_squared_error(y_true, y_pred)
    
    def _binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy Loss
        
        Formula:
        L = -1/n × Σ[y×log(ŷ) + (1-y)×log(1-ŷ)]
        
        Why BCE for classification?
        - Penalizes confident wrong predictions heavily
        - Works well with sigmoid/tanh outputs
        - Convex for logistic regression
        """
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        n = len(y_true)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return loss
    
    def _mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error Loss
        
        Formula:
        L = 1/n × Σ(y - ŷ)²
        """
        return np.mean((y_true - y_pred) ** 2)
    
    # ========== TRAINING ==========
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'Perceptron':
        """
        Train the perceptron using gradient descent
        
        Algorithm:
        1. Forward pass: Compute predictions
        2. Compute loss
        3. Backward pass: Compute gradients
        4. Update weights and bias
        5. Repeat for n_iterations
        
        Gradient Derivation (for sigmoid + BCE):
        ────────────────────────────────────────
        
        Loss: L = -1/n × Σ[y×log(ŷ) + (1-y)×log(1-ŷ)]
        
        Where ŷ = σ(z) and z = X·w + b
        
        Using chain rule:
        ∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂w
        
        Step by step:
        1. ∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ) = (ŷ-y)/(ŷ(1-ŷ))
        2. ∂ŷ/∂z = ŷ(1-ŷ)  (sigmoid derivative)
        3. ∂z/∂w = X
        
        Therefore:
        ∂L/∂w = (ŷ-y)/(ŷ(1-ŷ)) × ŷ(1-ŷ) × X
              = (ŷ-y) × X
        
        Similarly:
        ∂L/∂b = ŷ - y
        
        This beautiful simplification is why we use sigmoid + BCE!
        """
        n_samples, n_features = X.shape
        
        # Reset training history
        self.loss_history = []
        self.accuracy_history = []
        
        for iteration in range(self.n_iterations):
            # ===== FORWARD PASS =====
            predictions = self.forward(X)
            
            # ===== COMPUTE LOSS =====
            loss = self._compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            # ===== COMPUTE ACCURACY =====
            if self.activation_name in ['sigmoid', 'tanh', 'step']:
                pred_labels = (predictions >= 0.5).astype(int)
            else:
                pred_labels = predictions
            
            accuracy = np.mean(pred_labels == y)
            self.accuracy_history.append(accuracy)
            
            # ===== BACKWARD PASS =====
            # For sigmoid + BCE or tanh + BCE:
            # Gradient simplifies to: error = predictions - y
            error = predictions - y
            
            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # ===== UPDATE WEIGHTS =====
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1:4d}/{self.n_iterations} | "
                      f"Loss: {loss:.6f} | Accuracy: {accuracy:.4f}")
        
        return self
    
    # ========== PREDICTION ==========
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.forward(X)
        
        if self.activation_name in ['sigmoid', 'tanh']:
            return (probabilities >= 0.5).astype(int)
        else:
            return probabilities
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Returns raw output from activation function
        """
        return self.forward(X)
    
    # ========== VISUALIZATION ==========
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)):
        """Plot loss and accuracy over training"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        axes[0].plot(self.loss_history, linewidth=2, color='blue')
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.accuracy_history, linewidth=2, color='green')
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot decision boundary for 2D data
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            2D input data
        y : ndarray of shape (n_samples,)
            True labels
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plotting only works for 2D data")
        
        plt.figure(figsize=figsize)
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02),
            np.arange(y_min, y_max, 0.02)
        )
        
        # Predict on mesh
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        contour = plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
        plt.colorbar(contour)
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', 
                             edgecolors='k', s=100, linewidth=1)
        
        plt.xlabel('X₁', fontsize=12)
        plt.ylabel('X₂', fontsize=12)
        plt.title(f'Perceptron Decision Boundary ({self.activation_name} activation)', 
                 fontsize=14, fontweight='bold')
        plt.legend(*scatter.legend_elements(), title='Class')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # ========== MODEL INFORMATION ==========
    
    def get_weights(self) -> Tuple[np.ndarray, float]:
        """Get learned weights and bias"""
        return self.weights.copy(), self.bias
    
    def get_training_history(self) -> Tuple[List[float], List[float]]:
        """Get training history"""
        return self.loss_history.copy(), self.accuracy_history.copy()
```

---

## Part 3: Complete Examples and Experiments

### Example 1: OR Gate (Linearly Separable)

```python
print("=" * 70)
print("PERCEPTRON: LEARNING THE OR GATE")
print("=" * 70)

# Training data for OR gate
X_or = np.array([
    [0, 0],  # Input: 0 OR 0
    [0, 1],  # Input: 0 OR 1
    [1, 0],  # Input: 1 OR 0
    [1, 1]   # Input: 1 OR 1
])

y_or = np.array([0, 1, 1, 1])  # Expected outputs

print("\nTraining Data:")
print("X (inputs)     y (output)")
for i in range(len(X_or)):
    print(f"{X_or[i]}  →  {y_or[i]}")

# Create perceptron with different activations
activations = ['sigmoid', 'tanh', 'relu']

for activation in activations:
    print(f"\n{'='*70}")
    print(f"Training with {activation.upper()} activation")
    print(f"{'='*70}")
    
    perceptron = Perceptron(
        input_size=2,
        learning_rate=0.1,
        n_iterations=1000,
        activation=activation
    )
    
    perceptron.fit(X_or, y_or, verbose=True)
    
    # Evaluate
    predictions = perceptron.predict(X_or)
    probabilities = perceptron.predict_proba(X_or)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Input    →  Probability  →  Prediction  →  Actual")
    print(f"{'-'*70}")
    
    for i in range(len(X_or)):
        print(f"{X_or[i]}  →  {probabilities[i]:.6f}  →  {predictions[i]}  →  {y_or[i]}")
    
    accuracy = np.mean(predictions == y_or)
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    
    # Visualize
    perceptron.plot_training_history()
    perceptron.plot_decision_boundary(X_or, y_or)
```

### Example 2: AND Gate

```python
print("\n" + "=" * 70)
print("PERCEPTRON: LEARNING THE AND GATE")
print("=" * 70)

y_and = np.array([0, 0, 0, 1])  # AND gate outputs

perceptron_and = Perceptron(
    input_size=2,
    learning_rate=0.1,
    n_iterations=1000,
    activation='sigmoid'
)

perceptron_and.fit(X_or, y_and, verbose=True)

predictions_and = perceptron_and.predict(X_or)
print(f"\nPredictions: {predictions_and}")
print(f"Actual:      {y_and}")
print(f"Accuracy:    {np.mean(predictions_and == y_and):.4f}")
```

### Example 3: XOR Problem (The Limitation)

```python
print("\n" + "=" * 70)
print("PERCEPTRON: THE XOR PROBLEM (EXPECTED TO FAIL)")
print("=" * 70)

y_xor = np.array([0, 1, 1, 0])  # XOR gate outputs

print("\nXOR Truth Table:")
print("X (inputs)     y (output)")
for i in range(len(X_or)):
    print(f"{X_or[i]}  →  {y_xor[i]}")

print("\n💡 KEY INSIGHT: XOR is NOT linearly separable!")
print("   No single straight line can separate class 0 from class 1")
print("   This limitation led to the 'AI Winter' in the 1970s")
print("   Solution: Multiple layers (deep learning)!\n")

perceptron_xor = Perceptron(
    input_size=2,
    learning_rate=0.1,
    n_iterations=2000,
    activation='sigmoid'
)

perceptron_xor.fit(X_or, y_xor, verbose=True)

predictions_xor = perceptron_xor.predict(X_or)
print(f"\nFinal Predictions: {predictions_xor}")
print(f"Actual:            {y_xor}")
print(f"Accuracy:          {np.mean(predictions_xor == y_xor):.4f}")

print("\n📊 As you can see, single-layer perceptron CANNOT solve XOR!")
print("   We need Multi-Layer Perceptrons (MLPs)!")
```

---

## Part 4: Practice Problems

### Level 1: Basic Understanding

**Problem 1.1: NOT Gate**
```python
"""
Implement perceptron for NOT gate:
- Input: [0] → Output: [1]
- Input: [1] → Output: [0]

Questions:
1. Can a single perceptron solve this? Why or why not?
2. If yes, what's the decision boundary?
3. Implement and verify

Hint: NOT gate has only 1 input feature!
"""

# Your code here
X_not = np.array([[0], [1]])
y_not = np.array([1, 0])

# Create and train perceptron
# Test and visualize
```

**Problem 1.2: Activation Function Derivatives**
```python
"""
Derive the derivatives of:
1. Sigmoid: σ(x) = 1/(1+e^(-x))
2. Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
3. ReLU: ReLU(x) = max(0, x)

Show all steps of your derivation.
Verify your derivatives numerically.
"""

# Numerical verification template
def numerical_derivative(func, x, h=1e-5):
    """Compute numerical derivative"""
    return (func(x + h) - func(x - h)) / (2 * h)

# Compare with your analytical derivatives
```

### Level 2: Implementation

**Problem 2.1: Perceptron with Momentum**
```python
"""
Add momentum to the perceptron update rule:

Standard update:
w = w - learning_rate × dw

With momentum:
v = momentum × v - learning_rate × dw
w = w + v

Where:
- v: Velocity (accumulates past gradients)
- momentum: Typically 0.9

Tasks:
1. Implement PerceptronWithMomentum class
2. Compare convergence with standard perceptron
3. Experiment with different momentum values (0.5, 0.9, 0.99)
4. Plot learning curves
"""

class PerceptronWithMomentum(Perceptron):
    def __init__(self, input_size, learning_rate=0.01, momentum=0.9, n_iterations=1000):
        super().__init__(input_size, learning_rate, n_iterations)
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None
    
    def fit(self, X, y, verbose=True):
        # Initialize velocity
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = 0.0
        
        # Your implementation here
        pass
```

**Problem 2.2: Multi-class Perceptron**
```python
"""
Extend perceptron for multi-class classification using One-vs-Rest:

For K classes:
- Train K binary perceptrons
- Perceptron k distinguishes class k from all others
- Predict class with highest confidence score

Tasks:
1. Implement MultiClassPerceptron class
2. Test on Iris dataset (3 classes)
3. Compare with scikit-learn's Perceptron
4. Visualize decision boundaries
"""

class MultiClassPerceptron:
    def __init__(self, n_classes, input_size, learning_rate=0.01, n_iterations=1000):
        self.n_classes = n_classes
        self.perceptrons = []
        
        # Create one perceptron per class
        for i in range(n_classes):
            self.perceptrons.append(
                Perceptron(input_size, learning_rate, n_iterations)
            )
    
    def fit(self, X, y):
        # Train each perceptron for its class
        pass
    
    def predict(self, X):
        # Return class with highest score
        pass
```

### Level 3: Advanced

**Problem 3.1: Kernel Perceptron**
```python
"""
Implement the Kernel Perceptron algorithm:

The kernel trick allows perceptron to learn non-linear boundaries:

1. Map input x to higher-dimensional space: φ(x)
2. Use kernel function: K(x, x') = φ(x)·φ(x')
3. Common kernels:
   - Linear: K(x, x') = x·x'
   - Polynomial: K(x, x') = (x·x' + c)^d
   - RBF: K(x, x') = exp(-γ||x-x'||²)

Tasks:
1. Implement KernelPerceptron class
2. Test on XOR problem (should work now!)
3. Compare different kernels
4. Visualize decision boundaries
"""

class KernelPerceptron:
    def __init__(self, kernel='rbf', gamma=1.0, degree=3, n_iterations=1000):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.n_iterations = n_iterations
        self.support_vectors = []
        self.alphas = []
    
    def _kernel_function(self, X1, X2):
        """Compute kernel matrix"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + 1) ** self.degree
        elif self.kernel == 'rbf':
            # RBF kernel implementation
            pass
    
    def fit(self, X, y):
        # Kernel perceptron training algorithm
        pass
    
    def predict(self, X):
        # Prediction using support vectors
        pass
```

**Problem 3.2: Averaged Perceptron**
```python
"""
Implement the Averaged Perceptron algorithm:

Standard perceptron can oscillate near the solution.
Averaged perceptron averages all weight vectors seen during training.

Algorithm:
1. Train standard perceptron
2. Keep running sum of all weight vectors
3. Final weights = average of all weights

Tasks:
1. Implement AveragedPerceptron class
2. Compare with standard perceptron on:
   - Convergence speed
   - Final accuracy
   - Stability
3. Test on noisy dataset
"""

class AveragedPerceptron(Perceptron):
    def __init__(self, input_size, learning_rate=0.01, n_iterations=1000):
        super().__init__(input_size, learning_rate, n_iterations)
        self.sum_weights = np.zeros(input_size)
        self.sum_bias = 0.0
        self.count = 0
    
    def fit(self, X, y, verbose=True):
        # Your implementation
        # Average weights at the end
        pass
```

---

## 📊 Summary Tables

### Activation Functions Comparison

| Function | Formula | Range | Zero-Centered | Use Case |
|----------|---------|-------|---------------|----------|
| Step | 1 if z≥0, else 0 | {0, 1} | No | Original perceptron |
| Sigmoid | 1/(1+e⁻ᶻ) | (0, 1) | No | Binary output |
| Tanh | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (-1, 1) | Yes | Hidden layers |
| ReLU | max(0, z) | [0, ∞) | No | Default hidden |

### Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| MSE | (y-ŷ)² | Regression |
| BCE | -[y·log(ŷ)+(1-y)·log(1-ŷ)] | Binary classification |
| Hinge | max(0, 1-y·ŷ) | SVM |

### Learning Rate Guidelines

| Range | Effect | Recommendation |
|-------|--------|----------------|
| 0.001 | Very slow | Safe but slow |
| 0.01 | Moderate | Good default |
| 0.1 | Fast | May oscillate |
| >0.1 | Too fast | Likely to diverge |

---

## 📝 Notes Section

### Key Concepts Summary:


### Common Mistakes to Avoid:


### Code Snippets to Remember:


---

**Status:** ✅ Complete
**Next:** Multi-Layer Perceptron (MLP)
