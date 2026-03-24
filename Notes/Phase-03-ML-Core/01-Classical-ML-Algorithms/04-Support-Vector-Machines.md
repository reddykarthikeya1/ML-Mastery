# 7.4 Support Vector Machines

## 🎯 Learning Objectives
After completing this section, you will master:
1. **SVM Fundamentals**: Understand maximum margin classification and support vectors
2. **Kernel Trick**: Master linear, polynomial, RBF, and sigmoid kernels
3. **Soft vs Hard Margin**: Learn regularization through the C parameter
4. **Support Vector Regression**: Apply SVM principles to regression problems
5. **Practical Implementation**: Build SVM from scratch and tune hyperparameters

---

#### 🧒 ELI5: SVM, Maximum Margin & Kernel Trick

> Imagine you're separating red and blue marbles on a table.
>
> **SVM** (Finding the best dividing line):
>
> **Simple approach**: Draw ANY line between red and blue
> - Problem: Line is RIGHT NEXT to marbles
> - New marble rolls in → Might be on wrong side!
>
> **SVM approach**: Draw the WIDEST possible street!
> - Street edges touch closest red and blue marbles
> - Those closest marbles = "Support Vectors"
> - Wide street = More room for error!
> - New marble → Much more likely to be on correct side!
>
> **Support Vectors** (The important marbles):
> - Only the CLOSEST marbles matter
> - Move far-away marbles? Line doesn't change!
> - Move support vectors? Line moves!
> - That's why it's called "Support" Vector Machine!
>
> **Kernel Trick** (When marbles are mixed up):
>
> **Problem**: Red and blue marbles are COMPLETELY mixed!
> - No straight line can separate them
> - Red in center, blue in circle around it
>
> **Kernel Trick** (Lift to 3D!):
> - Imagine a magic wand that makes marbles JUMP up!
> - Red marbles jump HIGH
> - Blue marbles jump LOW
> - Now you can put a FLAT BOARD between high and low!
> - Look from above: Board looks like a CIRCLE!
>
> **Different Kernels** (Different magic tricks):
> - **Linear**: No magic, straight line only
> - **Polynomial**: "Make a curved boundary"
> - **RBF** (Most common): "Complex wiggly boundary"
> - Like: Choosing different shaped cutting tools!
>
> **Soft vs Hard Margin** (Allowing some mistakes):
>
> **Hard Margin**: "NO marbles can be on wrong side!"
> - One weird outlier? Street becomes SUPER narrow!
> - Overfits to outliers
>
> **Soft Margin** (C parameter): "A few mistakes are OK"
> - C = 100: "Almost no mistakes allowed!" (narrow street)
> - C = 0.01: "Some mistakes are fine" (wide street)
> - Like: "Is it worth making street narrower to catch that ONE weird marble?"

</details>

---

## 📚 SVM Fundamentals

### 7.4.1 Maximum Margin Classifier

**Core Idea:** Find the optimal hyperplane that separates classes with the **maximum margin**.

**Visual Intuition:**
```
Class A: ×    |    Class B: ○

    ×         |         ○
      ×       |       ○
        ×     |     ○
----------×---|---○----------  ← Decision Boundary (w·x + b = 0)
            × | ○
    Margin → ||| ← Margin
      ×       |       ○
              |
Support       |    Support
Vectors (×)   |    Vectors (○)
```

**Key Concepts:**
- **Hyperplane**: Decision boundary (w·x + b = 0)
- **Margin**: Distance between hyperplane and nearest data points
- **Support Vectors**: Data points closest to the hyperplane
- **Optimal Hyperplane**: Maximizes the margin

### Mathematical Formulation

**Hyperplane Equation:**
$$w^T x + b = 0$$

Where:
- $w$: Weight vector (perpendicular to hyperplane)
- $b$: Bias term
- $x$: Input feature vector

**Decision Function:**
$$f(x) = \begin{cases} 
+1 & \text{if } w^T x + b \geq 0 \\
-1 & \text{if } w^T x + b < 0 
\end{cases}$$

**Margin Calculation:**
```
Distance from point x_i to hyperplane = |w^T x_i + b| / ||w||

For support vectors: |w^T x_i + b| = 1

Therefore, margin = 2 / ||w||
```

**Optimization Problem (Hard Margin):**
```
Minimize: ½||w||²
Subject to: y_i(w^T x_i + b) ≥ 1 for all i

Goal: Minimize ||w|| (maximize margin) while correctly classifying all points
```

### 7.4.2 Hard Margin SVM

**Assumptions:**
- Data is linearly separable
- No noise or outliers
- Perfect classification required

**Primal Formulation:**
$$\min_{w,b} \frac{1}{2}||w||^2$$
$$\text{subject to } y_i(w^T x_i + b) \geq 1, \forall i$$

**Lagrangian Formulation:**
$$L(w,b,\alpha) = \frac{1}{2}||w||^2 - \sum_{i=1}^{n} \alpha_i [y_i(w^T x_i + b) - 1]$$

Where $\alpha_i \geq 0$ are Lagrange multipliers.

**KKT Conditions:**
1. Stationarity: $\frac{\partial L}{\partial w} = 0$, $\frac{\partial L}{\partial b} = 0$
2. Primal feasibility: $y_i(w^T x_i + b) \geq 1$
3. Dual feasibility: $\alpha_i \geq 0$
4. Complementary slackness: $\alpha_i[y_i(w^T x_i + b) - 1] = 0$

### 7.4.3 Soft Margin SVM

**Problem with Hard Margin:**
- Real data is rarely perfectly separable
- Sensitive to outliers
- Can lead to overfitting

**Solution:** Introduce slack variables $\xi_i$

**Soft Margin Formulation:**
$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} \xi_i$$
$$\text{subject to: } y_i(w^T x_i + b) \geq 1 - \xi_i, \xi_i \geq 0$$

**Hinge Loss:**
$$L(y, f(x)) = \max(0, 1 - y \cdot f(x))$$

```
Loss
  ↑
  |     /
  |    /
  |   /
  |  /
  | /
  |/___________→ y·f(x)
  0    1

When y·f(x) ≥ 1: Loss = 0 (correctly classified with margin)
When y·f(x) < 1: Loss = 1 - y·f(x) (penalized)
```

**C Parameter (Regularization):**
| C Value | Effect | Use Case |
|---------|--------|----------|
| Small C | Large margin, more misclassifications | Noisy data, prevent overfitting |
| Large C | Small margin, fewer misclassifications | Clean data, complex boundaries |
| C → ∞ | Approaches hard margin | Perfectly separable data |

---

## 📚 Kernel Functions

### 7.4.4 The Kernel Trick

**Problem:** Data is not linearly separable in original space

**Solution:** Map to higher-dimensional space where it becomes separable

**Kernel Trick:** Compute dot products in high-dimensional space without explicitly mapping

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

### Common Kernel Functions

**1. Linear Kernel**
```python
def linear_kernel(x1, x2):
    """K(x1, x2) = x1 · x2"""
    return np.dot(x1, x2)
```
- Use when: Data is linearly separable
- Parameters: None
- Fast computation

**2. Polynomial Kernel**
```python
def polynomial_kernel(x1, x2, degree=3, coef0=1, gamma=None):
    """K(x1, x2) = (γ * x1 · x2 + coef0)^degree"""
    if gamma is None:
        gamma = 1.0 / len(x1)
    return (gamma * np.dot(x1, x2) + coef0) ** degree
```
- Use when: Features interact
- Parameters: degree (d), coef0 (r), gamma (γ)
- Can capture complex patterns

**3. RBF (Gaussian) Kernel** (Most Popular)
```python
def rbf_kernel(x1, x2, gamma=None):
    """K(x1, x2) = exp(-γ * ||x1 - x2||²)"""
    if gamma is None:
        gamma = 1.0 / len(x1)
    sq_dist = np.sum((x1 - x2) ** 2)
    return np.exp(-gamma * sq_dist)
```
- Use when: Unknown decision boundary (default choice)
- Parameters: gamma (γ)
- Maps to infinite dimensions
- Local influence (points far apart have ~0 similarity)

**4. Sigmoid Kernel**
```python
def sigmoid_kernel(x1, x2, gamma=None, coef0=0):
    """K(x1, x2) = tanh(γ * x1 · x2 + coef0)"""
    if gamma is None:
        gamma = 1.0 / len(x1)
    return np.tanh(gamma * np.dot(x1, x2) + coef0)
```
- Use when: Similar to neural networks
- Parameters: gamma, coef0
- Not always positive semi-definite

### Kernel Comparison

| Kernel | Formula | Parameters | Best For |
|--------|---------|------------|----------|
| Linear | x₁·x₂ | None | High-dimensional, linear data |
| Polynomial | (γx₁·x₂ + r)^d | degree, gamma, coef0 | Feature interactions |
| RBF | exp(-γ\|\|x₁-x₂\|\|²) | gamma | General purpose (default) |
| Sigmoid | tanh(γx₁·x₂ + r) | gamma, coef0 | Neural network-like |

**Gamma Parameter (γ) in RBF:**
```
Small gamma (γ → 0):
- Far-reaching influence
- Smooth decision boundary
- Low variance, high bias

Large gamma (γ → ∞):
- Local influence only
- Complex, wiggly boundary
- High variance, low bias
```

---

## 📚 Dual Formulation and SMO

### 7.4.5 Primal vs Dual

**Primal Problem:**
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} \xi_i$$

**Dual Problem:**
$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
$$\text{subject to: } 0 \leq \alpha_i \leq C, \sum_{i=1}^{n} \alpha_i y_i = 0$$

**Advantages of Dual:**
- Only involves dot products (kernel trick)
- Sparse solution (only support vectors have αᵢ > 0)
- Easier to optimize

**Recovering w and b:**
$$w = \sum_{i=1}^{n} \alpha_i y_i \phi(x_i)$$
$$b = y_k - \sum_{i=1}^{n} \alpha_i y_i K(x_i, x_k) \text{ for any support vector } k$$

### 7.4.6 SMO Algorithm (Sequential Minimal Optimization)

**Idea:** Break large QP problem into smallest possible sub-problems

**SMO Steps:**
```
1. Initialize α = 0
2. Repeat until convergence:
   a. Select two Lagrange multipliers α_i, α_j
   b. Optimize these two while keeping others fixed
   c. Update threshold b
   d. Check KKT conditions
3. Return α, b
```

**Why Two Variables?**
- One variable: Constrained by Σαᵢyᵢ = 0
- Two variables: Can optimize analytically
- Simple box constraints: 0 ≤ αᵢ ≤ C

**SMO Update Rules:**
```python
def smo_step(alpha_i, alpha_j, y_i, y_j, K_ij, E_i, E_j, C):
    """
    One SMO optimization step
    
    E_i = f(x_i) - y_i (prediction error)
    """
    # Compute bounds
    if y_i != y_j:
        L = max(0, alpha_j - alpha_i)
        H = min(C, C + alpha_j - alpha_i)
    else:
        L = max(0, alpha_i + alpha_j - C)
        H = min(C, alpha_i + alpha_j)
    
    if L == H:
        return alpha_i, alpha_j
    
    # Compute eta
    eta = 2 * K_ij - K_ii - K_jj
    
    if eta >= 0:
        return alpha_i, alpha_j
    
    # Update alpha_j
    alpha_j_new = alpha_j - y_j * (E_i - E_j) / eta
    
    # Clip to [L, H]
    alpha_j_new = np.clip(alpha_j_new, L, H)
    
    # Update alpha_i
    alpha_i_new = alpha_i + y_i * y_j * (alpha_j - alpha_j_new)
    
    return alpha_i_new, alpha_j_new
```

---

## 📚 Support Vector Regression (SVR)

### 7.4.7 SVR Fundamentals

**Key Difference from SVM Classification:**
- SVM: Maximize margin between classes
- SVR: Fit data within ε-tube with minimum complexity

**ε-Insensitive Loss:**
$$L_\epsilon(y, f(x)) = \begin{cases}
0 & \text{if } |y - f(x)| \leq \epsilon \\
|y - f(x)| - \epsilon & \text{otherwise}
\end{cases}$$

**Visual Representation:**
```
y
↑
|     ○  ○
|   ○/│\○
|  ○─┼─┼─○  ← Upper bound: f(x) + ε
|   ○│\│○
|    ○\│○
|─────○\│──────  ← f(x) (prediction)
|     │\○
|     │ \
|     │  \
|     │   \
|     │    \
|─────┼─────┼────  ← Lower bound: f(x) - ε
|     │     │
|    ε│     │ε
|     │     │
+----------------→ x

Points inside ε-tube: No penalty
Points outside: Penalized linearly
```

**SVR Optimization:**
$$\min_{w,b,\xi,\xi^*} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} (\xi_i + \xi_i^*)$$

Subject to:
$$y_i - (w^T x_i + b) \leq \epsilon + \xi_i$$
$$(w^T x_i + b) - y_i \leq \epsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

### SVR Implementation

```python
import numpy as np

class SupportVectorRegression:
    """
    Support Vector Regression using kernel trick.
    Simplified implementation for educational purposes.
    """
    
    def __init__(self, epsilon=0.1, C=1.0, kernel='rbf', gamma='scale'):
        self.epsilon = epsilon
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.support_vectors = None
        self.sv_coefficients = None
        self.dual_coefficients = None
        self.bias = 0
    
    def _rbf_kernel(self, X1, X2):
        """RBF kernel matrix"""
        if self.gamma == 'scale':
            gamma = 1.0 / (X1.shape[1] * X1.var())
        else:
            gamma = self.gamma
        
        # Compute squared Euclidean distances
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sq_dists = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        
        return np.exp(-gamma * sq_dists)
    
    def _linear_kernel(self, X1, X2):
        """Linear kernel matrix"""
        return np.dot(X1, X2.T)
    
    def _compute_kernel(self, X1, X2):
        """Compute kernel matrix based on selected kernel"""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def fit(self, X, y, max_iter=1000, tol=1e-3):
        """
        Fit SVR model using simplified coordinate descent.
        
        Note: This is a simplified version. Production implementations
        use SMO or other specialized QP solvers.
        """
        n_samples = len(X)
        
        # Initialize dual coefficients
        self.alpha = np.zeros(n_samples)  # For positive slack
        self.alpha_star = np.zeros(n_samples)  # For negative slack
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        
        # Simplified coordinate descent
        for iteration in range(max_iter):
            alpha_prev = self.alpha.copy()
            alpha_star_prev = self.alpha_star.copy()
            
            for i in range(n_samples):
                # Compute prediction
                f_i = np.sum((self.alpha - self.alpha_star) * K[:, i]) + self.bias
                
                # Compute error
                E_i = f_i - y[i]
                
                # Update alpha (for y_i - f(x_i) > ε)
                if (y[i] - f_i > self.epsilon and self.alpha[i] < self.C) or \
                   (y[i] - f_i < -self.epsilon and self.alpha[i] > 0):
                    
                    # Gradient
                    grad = y[i] - f_i - self.epsilon
                    
                    # Update
                    self.alpha[i] += grad / (K[i, i] + 1e-8)
                    self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
                
                # Update alpha_star (for f(x_i) - y_i > ε)
                if (f_i - y[i] > self.epsilon and self.alpha_star[i] < self.C) or \
                   (f_i - y[i] < -self.epsilon and self.alpha_star[i] > 0):
                    
                    grad = f_i - y[i] - self.epsilon
                    self.alpha_star[i] += grad / (K[i, i] + 1e-8)
                    self.alpha_star[i] = np.clip(self.alpha_star[i], 0, self.C)
            
            # Check convergence
            if (np.max(np.abs(self.alpha - alpha_prev)) < tol and 
                np.max(np.abs(self.alpha_star - alpha_star_prev)) < tol):
                break
        
        # Find support vectors (non-zero coefficients)
        sv_threshold = 1e-5
        sv_mask = (np.abs(self.alpha) > sv_threshold) | \
                  (np.abs(self.alpha_star) > sv_threshold)
        
        self.support_vectors = X[sv_mask]
        self.dual_coefficients = (self.alpha - self.alpha_star)[sv_mask]
        
        # Compute bias using support vectors on margin
        margin_sv_mask = ((self.alpha > sv_threshold) & (self.alpha < self.C - 1e-5)) | \
                         ((self.alpha_star > sv_threshold) & (self.alpha_star < self.C - 1e-5))
        
        if np.any(margin_sv_mask):
            sv_indices = np.where(margin_sv_mask)[0]
            self.bias = np.mean([
                y[i] - np.sum((self.alpha - self.alpha_star) * K[:, i])
                for i in sv_indices
            ])
        
        return self
    
    def predict(self, X):
        """Predict using trained SVR model"""
        K = self._compute_kernel(X, self.support_vectors)
        return np.dot(K, self.dual_coefficients) + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
```

---

## 💻 Implementation from Scratch: SVM Classifier

```python
import numpy as np
from typing import Tuple, Optional

class SVMClassifier:
    """
    Support Vector Machine Classifier implemented from scratch.
    Uses the dual formulation with kernel trick.
    Simplified gradient-based optimization for educational purposes.
    """
    
    def __init__(self, 
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 gamma: Optional[float] = None,
                 degree: int = 3,
                 tol: float = 1e-3,
                 max_iter: int = 1000):
        """
        Initialize SVM Classifier.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: Kernel coefficient (auto-calculated if None)
            degree: Degree for polynomial kernel
            tol: Tolerance for convergence
            max_iter: Maximum iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        
        self.support_vectors = None
        self.sv_labels = None
        self.dual_coefficients = None
        self.bias = 0
        self.alphas = None
    
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Linear kernel: K(x, x') = x · x'"""
        return np.dot(X1, X2.T)
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel: K(x, x') = exp(-γ||x - x'||²)"""
        if self.gamma is None:
            gamma = 1.0 / (X1.shape[1] * X1.var())
        else:
            gamma = self.gamma
        
        # Efficient computation of squared distances
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sq_dists = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        
        # Ensure non-negative (numerical stability)
        sq_dists = np.maximum(sq_dists, 0)
        
        return np.exp(-gamma * sq_dists)
    
    def _poly_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Polynomial kernel: K(x, x') = (γx · x' + coef0)^degree"""
        if self.gamma is None:
            gamma = 1.0 / (X1.shape[1] * X1.var())
        else:
            gamma = self.gamma
        
        coef0 = 1.0
        return (gamma * np.dot(X1, X2.T) + coef0) ** self.degree
    
    def _sigmoid_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Sigmoid kernel: K(x, x') = tanh(γx · x' + coef0)"""
        if self.gamma is None:
            gamma = 1.0 / (X1.shape[1] * X1.var())
        else:
            gamma = self.gamma
        
        coef0 = 0.0
        return np.tanh(gamma * np.dot(X1, X2.T) + coef0)
    
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix based on selected kernel type"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == 'poly':
            return self._poly_kernel(X1, X2)
        elif self.kernel == 'sigmoid':
            return self._sigmoid_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        Fit SVM using simplified gradient ascent on dual problem.
        
        Dual problem:
        max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        s.t. 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0
        """
        n_samples = len(X)
        y_signed = np.where(y == y[0], 1, -1)  # Convert to {-1, +1}
        
        # Initialize alphas
        self.alphas = np.zeros(n_samples)
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        
        # Gradient ascent on dual
        for iteration in range(self.max_iter):
            alphas_prev = self.alphas.copy()
            
            for i in range(n_samples):
                # Compute gradient of dual objective
                gradient = 1 - y_signed[i] * np.sum(
                    self.alphas * y_signed * K[:, i]
                )
                
                # Gradient ascent step
                self.alphas[i] += self.tol * gradient
                
                # Project to feasible region [0, C]
                self.alphas[i] = np.clip(self.alphas[i], 0, self.C)
            
            # Check convergence
            if np.max(np.abs(self.alphas - alphas_prev)) < self.tol:
                break
        
        # Find support vectors (alpha > 0)
        sv_threshold = 1e-5
        sv_mask = self.alphas > sv_threshold
        
        self.support_vectors = X[sv_mask]
        self.sv_labels = y_signed[sv_mask]
        self.dual_coefficients = self.alphas[sv_mask] * y_signed[sv_mask]
        
        # Compute bias using support vectors on margin
        # (0 < alpha < C)
        margin_mask = (self.alphas > sv_threshold) & (self.alphas < self.C - 1e-5)
        
        if np.any(margin_mask):
            margin_indices = np.where(margin_mask)[0]
            K_margin = self._compute_kernel(X[margin_indices], self.support_vectors)
            predictions = np.dot(K_margin, self.dual_coefficients)
            self.bias = np.mean(y_signed[margin_indices] - predictions)
        else:
            # Fallback: use all support vectors
            K_sv = self._compute_kernel(self.support_vectors, self.support_vectors)
            predictions = np.dot(K_sv, self.dual_coefficients)
            self.bias = np.mean(self.sv_labels - predictions)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        decision_values = self.decision_function(X)
        return np.sign(decision_values)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function (distance from hyperplane)"""
        K = self._compute_kernel(X, self.support_vectors)
        return np.dot(K, self.dual_coefficients) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        y_pred = self.predict(X)
        y_true = np.where(y == y[0], 1, -1)
        return np.mean(y_pred == y_true)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates using Platt scaling approximation.
        P(y=1|x) ≈ 1 / (1 + exp(A·f(x) + B))
        """
        decision_values = self.decision_function(X)
        
        # Simplified Platt scaling (A=-1, B=0 for demonstration)
        probas = 1 / (1 + np.exp(-decision_values))
        
        # Return as 2D array [P(class_0), P(class_1)]
        return np.column_stack([1 - probas, probas])


# Example usage and comparison with sklearn
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    
    # Generate data
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=8,
        n_redundant=2, random_state=42
    )
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Custom SVM
    custom_svm = SVMClassifier(C=1.0, kernel='rbf', gamma='scale')
    custom_svm.fit(X_train_scaled, y_train)
    custom_acc = custom_svm.score(X_test_scaled, y_test)
    
    # sklearn SVM
    sklearn_svm = SVC(C=1.0, kernel='rbf', gamma='scale')
    sklearn_svm.fit(X_train_scaled, y_train)
    sklearn_acc = sklearn_svm.score(X_test_scaled, y_test)
    
    print(f"Custom SVM Accuracy: {custom_acc:.4f}")
    print(f"sklearn SVM Accuracy: {sklearn_acc:.4f}")
    
    # Compare kernels
    print("\nKernel Comparison:")
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        svm = SVMClassifier(C=1.0, kernel=kernel)
        svm.fit(X_train_scaled, y_train)
        acc = svm.score(X_test_scaled, y_test)
        print(f"  {kernel}: {acc:.4f}")
```

---

## 📊 Summary Tables

### SVM Hyperparameters

| Hyperparameter | Description | Typical Values | Impact |
|----------------|-------------|----------------|--------|
| C | Regularization parameter | 0.001, 0.01, 0.1, 1, 10, 100 | Controls margin vs misclassification tradeoff |
| kernel | Kernel function | linear, rbf, poly, sigmoid | Determines decision boundary shape |
| gamma | RBF/Poly/Sigmoid coefficient | scale, auto, 0.001-1 | Influence radius of training points |
| degree | Polynomial degree | 2, 3, 4 | Complexity of polynomial boundary |
| coef0 | Independent term in poly/sigmoid | 0, 1 | Offset in kernel function |

### Kernel Selection Guide

| Scenario | Recommended Kernel | Reason |
|----------|-------------------|--------|
| High-dimensional data (n_features > n_samples) | Linear | Data likely linearly separable |
| Unknown decision boundary | RBF (default) | Flexible, works well in practice |
| Text classification | Linear | High-dimensional, sparse features |
| Image classification | RBF | Complex non-linear boundaries |
| Feature interactions important | Polynomial | Captures feature combinations |
| Neural network-like behavior | Sigmoid | Similar activation pattern |

### SVM vs Other Classifiers

| Aspect | SVM | Logistic Regression | Random Forest |
|--------|-----|---------------------|---------------|
| Decision Boundary | Maximum margin | Probabilistic | Ensemble of trees |
| Kernel Trick | ✅ Yes | ❌ No | ❌ No |
| Outlier Sensitivity | Medium (depends on C) | High | Low |
| Probability Output | ❌ No (needs calibration) | ✅ Yes | ✅ Yes |
| Multi-class | One-vs-One/Rest | Native | Native |
| Training Speed | O(n²) to O(n³) | O(n·d) | O(n·log(n)) |
| Prediction Speed | O(#SV · d) | O(d) | O(#trees · depth) |

---

## 🎯 ML Applications

| Application | Description | Key SVM Feature |
|-------------|-------------|-----------------|
| **Text Classification** | Spam detection, sentiment analysis | High-dimensional linear SVM |
| **Image Recognition** | Face detection, object classification | RBF kernel for complex patterns |
| **Bioinformatics** | Protein classification, gene expression | Handles small samples well |
| **Handwriting Recognition** | Digit classification (MNIST) | Effective in high dimensions |
| **Financial Prediction** | Stock movement, credit scoring | Robust to overfitting |
| **Anomaly Detection** | One-class SVM for novelty detection | One-class formulation |

---

## 📝 Practice Problems

### Level 1: Basic

1. **Conceptual**: Explain the difference between hard margin and soft margin SVM
2. **Calculation**: Given w = [2, 3] and b = -1, calculate the margin width
3. **Understanding**: What are support vectors and why are they important?
4. **Code**: Implement the hinge loss function and plot it
5. **Analysis**: How does the C parameter affect the decision boundary?

### Level 2: Intermediate

1. **Implementation**: Build an SVM classifier from scratch using the dual formulation
2. **Experiment**: Compare different kernels (linear, RBF, polynomial) on the same dataset
3. **Analysis**: Plot decision boundaries for different gamma values in RBF kernel
4. **SVR**: Implement Support Vector Regression and compare with linear regression
5. **Tuning**: Use grid search to find optimal C and gamma for a classification problem

### Level 3: Advanced

1. **SMO Algorithm**: Implement the full SMO algorithm for SVM training
2. **Multi-class SVM**: Implement one-vs-one and one-vs-rest multi-class SVM
3. **Research**: Implement One-Class SVM for anomaly detection
4. **Optimization**: Compare gradient descent, coordinate descent, and SMO for SVM training
5. **Project**: Build a complete SVM pipeline with custom kernels for a real-world dataset

---

## 🔗 Related Topics
- [[01-Linear-Models]] - Compare with logistic regression
- [[03-Instance-Based-Learning]] - Alternative non-parametric approach
- [[08-Regularization-Techniques]] - Understanding regularization in SVM
- [[07-Dimensionality-Reduction]] - Preprocessing for SVM

---

**Status:** ✅ Complete  
**Next:** [[05-Naive-Bayes]]
