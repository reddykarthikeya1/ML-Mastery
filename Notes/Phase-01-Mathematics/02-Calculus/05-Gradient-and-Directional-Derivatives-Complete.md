# 1.2.5 Gradient and Directional Derivatives

## 🎯 Quick Overview
- **Gradient**: Vector of all partial derivatives
- **Directional derivative**: Rate of change in any direction
- **Foundation for**: Gradient descent, optimization, neural network training

---

## 1. Gradient Vector

### Definition

For a scalar function f(x₁, x₂, ..., xₙ), the **gradient** is:

```
∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)

Or as a column vector:
        [ ∂f/∂x₁ ]
        [ ∂f/∂x₂ ]
∇f =    [   ...   ]
        [ ∂f/∂xₙ ]
```

**Notation:**
- ∇f (nabla f)
- grad f
- Both mean the same thing!

### 2D Example

```
f(x, y) = x² + y²

∂f/∂x = 2x
∂f/∂y = 2y

∇f = (2x, 2y)

At point (1, 2):
∇f(1,2) = (2, 4)
```

### 3D Example

```
f(x, y, z) = x²y + yz³ - xz

∂f/∂x = 2xy - z
∂f/∂y = x² + z³
∂f/∂z = 3yz² - x

∇f = (2xy - z, x² + z³, 3yz² - x)
```

---

## 2. Geometric Interpretation of Gradient

### Key Properties

1. **Direction**: Points in direction of steepest increase
2. **Magnitude**: Rate of increase in that direction
3. **Perpendicular**: Orthogonal to level curves/surfaces

### Visual (2D)

```
Level curves of f(x,y) = x² + y²:

         ___
      _/     \_
     /    ↑    \    Gradient vectors point
    |  ←  ·  →  |   outward, perpendicular
     \_  ↓  _/    to level curves
        ‾‾‾

At each point, ∇f points toward higher values
```

### Why Gradient Points Uphill

```
Consider small change Δx:

Δf ≈ ∇f · Δx  (first-order approximation)

To maximize Δf:
- Choose Δx in same direction as ∇f
- Then ∇f · Δx = |∇f| |Δx| (maximum when parallel)

Therefore: gradient direction = steepest ascent
```

---

## 3. Directional Derivatives

### Definition

The **directional derivative** of f in direction **u** (unit vector):

```
D_u f(x) = ∇f(x) · u

where u is a UNIT vector (\|u\| = 1)
```

### Computation

**Step 1:** Find gradient ∇f
**Step 2:** Normalize direction vector to get unit vector u
**Step 3:** Compute dot product

**Example:**
```
f(x, y) = x² + 2xy + y²

Find rate of change at (1, 1) in direction v = (3, 4)

Step 1: ∇f = (2x + 2y, 2x + 2y)
        ∇f(1,1) = (4, 4)

Step 2: \|v\| = √(3² + 4²) = 5
        u = (3/5, 4/5)

Step 3: D_u f = ∇f · u
              = (4, 4) · (3/5, 4/5)
              = 12/5 + 16/5 = 28/5 = 5.6
```

### Interpretation

| Value | Meaning |
|-------|---------|
| D_u f > 0 | Function increasing in direction u |
| D_u f < 0 | Function decreasing in direction u |
| D_u f = 0 | No change in direction u (along level curve) |

---

## 4. Maximum Rate of Increase

### Theorem

The **maximum** value of D_u f is \|∇f\|, achieved when u points in the direction of ∇f.

**Proof:**
```
D_u f = ∇f · u = |∇f| |u| cos(θ) = |∇f| cos(θ)

Maximum when cos(θ) = 1, i.e., θ = 0
This means u is parallel to ∇f

Maximum value = |∇f|
```

### Example

```
f(x, y) = x² + y²
∇f(1, 2) = (2, 4)

Maximum rate of increase = |∇f| = √(4 + 16) = √20 ≈ 4.47

Direction of maximum increase = (2, 4) or normalized: (1/√5, 2/√5)
```

---

## 5. Level Curves and Level Surfaces

### Level Curves (2D)

For f(x, y), a **level curve** is:
```
{(x, y) : f(x, y) = c}  for constant c
```

**Example:** f(x, y) = x² + y²
```
Level curves are circles:
c = 1: x² + y² = 1  (unit circle)
c = 4: x² + y² = 4  (circle radius 2)
c = 9: x² + y² = 9  (circle radius 3)
```

### Level Surfaces (3D)

For f(x, y, z), a **level surface** is:
```
{(x, y, z) : f(x, y, z) = c}
```

**Example:** f(x, y, z) = x² + y² + z²
```
Level surfaces are spheres:
c = 1: x² + y² + z² = 1  (unit sphere)
c = 4: x² + y² + z² = 4  (sphere radius 2)
```

### Gradient is Perpendicular to Level Sets

**Theorem:**
```
∇f(x₀) is perpendicular to the level curve/surface through x₀
```

**Why:**
```
Along level curve: f(x, y) = c (constant)

For any tangent vector v to level curve:
D_v f = ∇f · v = 0

This means ∇f ⊥ v (orthogonal to all tangent vectors)
```

---

## 6. Applications to Optimization

### Gradient Descent

**Update rule:**
```
x_new = x_old - α · ∇f(x_old)

where α is the learning rate
```

**Intuition:**
- Move opposite to gradient (downhill)
- Step size controlled by α
- Repeat until convergence

**Visual:**
```
        Start
          ↓
     Follow -∇f
          ↓
     Follow -∇f
          ↓
    Minimum reached
```

### Why Gradient Descent Works

```
Taylor expansion:
f(x + Δx) ≈ f(x) + ∇f(x) · Δx

To minimize f(x + Δx):
Choose Δx = -α∇f(x)

Then: f(x + Δx) ≈ f(x) - α|∇f(x)|²

Since |∇f(x)|² ≥ 0, function decreases!
```

### Critical Points

**Classification using gradient:**

| Type | Condition |
|------|-----------|
| Critical point | ∇f = 0 |
| Minimum | ∇f = 0, Hessian positive definite |
| Maximum | ∇f = 0, Hessian negative definite |
| Saddle | ∇f = 0, Hessian indefinite |

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, gradient, hessian, lambdify
from mpl_toolkits.mplot3d import Axes3D

# === Symbolic Gradient ===

x, y, z = symbols('x y z')

# 2D function
f_2d = x**2 + 2*x*y + y**2
grad_2d = [diff(f_2d, var) for var in [x, y]]
print(f"f(x,y) = {f_2d}")
print(f"∇f = {grad_2d}")

# 3D function
f_3d = x**2*y + y*z**2 - x*z
grad_3d = [diff(f_3d, var) for var in [x, y, z]]
print(f"\nf(x,y,z) = {f_3d}")
print(f"∇f = {grad_3d}")

# === Numerical Gradient ===

def numerical_gradient(f, point, h=1e-5):
    """Compute gradient numerically"""
    point = np.array(point)
    grad = np.zeros_like(point)
    
    for i in range(len(point)):
        offset = np.zeros_like(point)
        offset[i] = h
        grad[i] = (f(point + offset) - f(point - offset)) / (2*h)
    
    return grad

# Example: f(x,y) = x² + y²
f = lambda x: x[0]**2 + x[1]**2
point = [1, 2]
grad = numerical_gradient(f, point)
print(f"\nNumerical gradient at {point}: {grad}")
print(f"Expected: [2, 4]")

# === Directional Derivative ===

def directional_derivative(f, point, direction):
    """Compute directional derivative"""
    point = np.array(point)
    direction = np.array(direction)
    
    # Normalize direction
    u = direction / np.linalg.norm(direction)
    
    # Compute gradient
    grad = numerical_gradient(f, point)
    
    # Directional derivative = gradient dot direction
    return np.dot(grad, u)

# Example
f = lambda x: x[0]**2 + 2*x[0]*x[1] + x[1]**2
point = [1, 1]
direction = [3, 4]

dd = directional_derivative(f, point, direction)
print(f"\nDirectional derivative at {point} in direction {direction}:")
print(f"Result: {dd:.4f}")

# === Gradient Descent Visualization ===

def gradient_descent(f, grad_f, start, learning_rate=0.1, n_iterations=50):
    """Perform gradient descent"""
    path = [np.array(start, dtype=float)]
    
    for _ in range(n_iterations):
        current = path[-1]
        grad = grad_f(current)
        next_point = current - learning_rate * grad
        path.append(next_point)
    
    return np.array(path)

# Example: f(x,y) = x² + y²
f = lambda x: x[0]**2 + x[1]**2
grad_f = lambda x: np.array([2*x[0], 2*x[1]])

# Run gradient descent
start = [-3, -2]
path = gradient_descent(f, grad_f, start, learning_rate=0.1, n_iterations=20)

# Create mesh for visualization
x_vals = np.linspace(-4, 4, 100)
y_vals = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2

# Plot
fig = plt.figure(figsize=(12, 5))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
ax1.plot(path[:, 0], path[:, 1], f(path), 'r-o', linewidth=2, markersize=5)
ax1.set_title('Gradient Descent Path (3D)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
ax2.plot(path[:, 0], path[:, 1], 'w-o', linewidth=2, markersize=5)
ax2.plot(path[0, 0], path[0, 1], 'go', markersize=15, label='Start')
ax2.plot(path[-1, 0], path[-1, 1], 'r*', markersize=20, label='End')
ax2.set_title('Gradient Descent Path (Contour)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.axis('equal')
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()

# === Gradient Field Visualization ===

def plot_gradient_field(f, x_range, y_range, density=20):
    """Plot function contours with gradient field"""
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    
    # Compute function values
    Z = f([X, Y])
    
    # Compute gradients
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(density):
        for j in range(density):
            grad = numerical_gradient(f, [X[i,j], Y[i,j]])
            U[i,j] = grad[0]
            V[i,j] = grad[1]
    
    # Normalize for visualization
    N = np.sqrt(U**2 + V**2)
    U_norm = U / N
    V_norm = V / N
    
    plt.figure(figsize=(10, 8))
    
    # Contours
    plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    
    # Gradient field (quiver)
    plt.quiver(X, Y, U_norm, V_norm, color='white', alpha=0.7, scale=30)
    
    plt.title('Function Contours with Gradient Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='f(x,y)')
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: f(x,y) = x² - y² (saddle)
f = lambda x: x[0]**2 - x[1]**2
plot_gradient_field(f, (-3, 3), (-3, 3))

# === Maximum Rate of Increase ===

def max_rate_of_increase(f, point):
    """Compute maximum rate of increase and direction"""
    grad = numerical_gradient(f, point)
    max_rate = np.linalg.norm(grad)
    direction = grad / max_rate if max_rate > 0 else grad
    
    return max_rate, direction

# Example
f = lambda x: x[0]**2 + 2*x[1]**2 + x[2]**2
point = [1, 2, 1]

max_rate, direction = max_rate_of_increase(f, point)
print(f"\nAt point {point}:")
print(f"Maximum rate of increase: {max_rate:.4f}")
print(f"Direction: {direction}")
```

---

## 📊 Summary Table

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Gradient** | ∇f = (∂f/∂x₁, ..., ∂f/∂xₙ) | Vector of partial derivatives |
| **Directional Derivative** | D_u f = ∇f · u | Rate of change in direction u |
| **Max Rate** | \|∇f\| | Maximum possible directional derivative |
| **Max Direction** | ∇f/\|∇f\| | Direction of steepest ascent |
| **Gradient Descent** | x - α∇f | Move opposite to gradient |

---

## 🎯 ML Applications

| Application | Gradient Concept |
|-------------|-----------------|
| **Gradient Descent** | Follow -∇(loss) to minimize |
| **Backpropagation** | Compute ∇(loss) wrt all weights |
| **Neural Network Training** | Iterative gradient updates |
| **Feature Importance** | Magnitude of gradient |
| **Adversarial Examples** | ∇(loss) wrt input |
| **Natural Gradient** | Gradient in information geometry |

**Neural Network Example:**
```
Loss L(w₁, w₂, ..., wₙ)

Gradient: ∇L = (∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ)

Update: wᵢ = wᵢ - α · ∂L/∂wᵢ

This is gradient descent!
```

---

## ❓ Quick Check Questions

1. What does the gradient vector represent?
2. Why is the gradient perpendicular to level curves?
3. How do you compute a directional derivative?
4. In what direction is the rate of increase maximum?
5. Why does gradient descent use minus sign: x - α∇f?
6. What happens to the gradient at a minimum?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Gradient represents:**
   - Vector of all partial derivatives
   - Direction of steepest increase
   - Magnitude = maximum rate of increase

2. **Gradient perpendicular to level curves:**
   - Along level curve, f is constant
   - Directional derivative along curve = 0
   - ∇f · tangent = 0 → orthogonal

3. **Directional derivative:**
   - D_u f = ∇f · u
   - Normalize direction, dot with gradient

4. **Maximum rate direction:**
   - Same direction as gradient
   - Rate = \|∇f\|

5. **Minus sign in gradient descent:**
   - Gradient points uphill
   - We want to go downhill (minimize)
   - So move in -∇f direction

6. **Gradient at minimum:**
   - ∇f = 0 (critical point)
   - No direction of increase/decrease

</details>
---

**Status:** ✅ Complete
**Next:** Chain Rule for Multivariable Functions
