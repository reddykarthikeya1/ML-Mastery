# 1.2.11 Advanced Topics for ML

## 🎯 Quick Overview
- **Taylor Series**: Function approximation using polynomials
- **Convex Optimization**: Finding global minima efficiently
- **Foundation for**: Optimization algorithms, convergence analysis, neural network training

---

## 1. Taylor Series and Taylor Polynomials

### Motivation

**Question:** How can we approximate complex functions?

**Answer:** Use polynomials! (They're easy to compute, differentiate, integrate)

### Taylor Polynomial (1D)

The **nth-degree Taylor polynomial** of f(x) centered at a:

```
Pₙ(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ... + f⁽ⁿ⁾(a)(x-a)ⁿ/n!

In sigma notation:
Pₙ(x) = Σₖ₌₀ⁿ f⁽ⁿ⁾(a)·(x-a)ᵏ/k!
```

**Taylor's Theorem:**
```
f(x) = Pₙ(x) + Rₙ(x)

where Rₙ(x) is the remainder (error)
```

### Common Taylor Series

**At a = 0 (Maclaurin Series):**

| Function | Taylor Series | Convergence |
|----------|--------------|-------------|
| eˣ | 1 + x + x²/2! + x³/3! + ... = Σ xⁿ/n! | All x |
| sin(x) | x - x³/3! + x⁵/5! - x⁷/7! + ... | All x |
| cos(x) | 1 - x²/2! + x⁴/4! - x⁶/6! + ... | All x |
| ln(1+x) | x - x²/2 + x³/3 - x⁴/4 + ... | -1 < x ≤ 1 |
| 1/(1-x) | 1 + x + x² + x³ + ... | \|x\| < 1 |

### Examples

**Example 1:** Taylor series of eˣ at a = 0

```
f(x) = eˣ
f'(x) = eˣ
f''(x) = eˣ
...
f⁽ⁿ⁾(x) = eˣ

At a = 0:
f(0) = 1, f'(0) = 1, f''(0) = 1, ...

Pₙ(x) = 1 + x + x²/2! + x³/3! + ... + xⁿ/n!
```

**Example 2:** Taylor series of sin(x) at a = 0

```
f(x) = sin(x)
f'(x) = cos(x)
f''(x) = -sin(x)
f'''(x) = -cos(x)
f⁽⁴⁾(x) = sin(x)  (pattern repeats!)

At a = 0:
f(0) = 0, f'(0) = 1, f''(0) = 0, f'''(0) = -1, ...

P₅(x) = x - x³/6 + x⁵/120
```

---

## 2. Multivariate Taylor Expansion ⭐ CRITICAL

### Definition

For f(x₁, x₂, ..., xₙ) near point **a**:

**First-order (Linear approximation):**
```
f(x) ≈ f(a) + ∇f(a) · (x - a)
```

**Second-order (Quadratic approximation):**
```
f(x) ≈ f(a) + ∇f(a) · (x - a) + (1/2)(x - a)ᵀ H(a) (x - a)

where H is the Hessian matrix
```

### Hessian Matrix

```
        [ ∂²f/∂x₁²    ∂²f/∂x₁∂x₂   ...   ∂²f/∂x₁∂xₙ ]
        | ∂²f/∂x₂∂x₁   ∂²f/∂x₂²    ...   ∂²f/∂x₂∂xₙ |
H(f) =  |    ...         ...      ...      ...     |
        [ ∂²f/∂xₙ∂x₁   ∂²f/∂xₙ∂x₂   ...   ∂²f/∂xₙ²  ]
```

**For 2 variables:**
```
        [ f_xx   f_xy ]
H(f) =  |            |
        [ f_yx   f_yy ]
```

### Example

```
f(x, y) = x² + 2xy + y²

First derivatives:
f_x = 2x + 2y
f_y = 2x + 2y

Second derivatives:
f_xx = 2,   f_xy = 2
f_yx = 2,   f_yy = 2

Hessian:
        [ 2   2 ]
H(f) =  [     ]
        [ 2   2 ]

Taylor expansion at (0, 0):
f(x,y) ≈ f(0,0) + [f_x, f_y]·[x,y] + (1/2)[x,y]·H·[x,y]ᵀ
       = 0 + [0,0]·[x,y] + (1/2)[x,y]·[[2,2],[2,2]]·[x,y]ᵀ
       = 0 + 0 + (1/2)(2x² + 4xy + 2y²)
       = x² + 2xy + y²  ✓ (exact for quadratic!)
```

---

## 3. Newton's Method for Optimization

### Newton's Method (Root Finding)

**Goal:** Find x where f(x) = 0

**Update rule:**
```
xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
```

### Newton's Method for Optimization

**Goal:** Find x where f'(x) = 0 (critical point)

**Update rule:**
```
xₙ₊₁ = xₙ - f'(xₙ)/f''(xₙ)
```

**Multivariate version:**
```
xₙ₊₁ = xₙ - [H(xₙ)]⁻¹ ∇f(xₙ)

where H is the Hessian matrix
```

### Comparison with Gradient Descent

| Method | Update Rule | Pros | Cons |
|--------|-------------|------|------|
| **Gradient Descent** | x - α∇f | Simple, scalable | Slow convergence |
| **Newton's Method** | x - H⁻¹∇f | Fast (quadratic) | Hessian expensive |

### When to Use Newton's Method

✅ Small to medium problems
✅ Hessian is easy to compute
✅ Hessian is positive definite
✅ Need fast convergence

❌ Very high dimensions
❌ Hessian is singular or ill-conditioned
❌ Hessian is expensive to compute

---

## 4. Convex Functions and Convex Optimization

### Convex Function (Definition)

A function f is **convex** if for all x, y and λ ∈ [0, 1]:

```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```

**Geometric meaning:** Line segment between any two points lies ABOVE the graph.

### Tests for Convexity

**1D:**
```
f is convex ⟺ f''(x) ≥ 0 for all x
f is strictly convex ⟺ f''(x) > 0 for all x
```

**Multivariate:**
```
f is convex ⟺ Hessian H(f) is positive semidefinite
f is strictly convex ⟺ Hessian H(f) is positive definite
```

### Common Convex Functions

| Function | Convex? | Notes |
|----------|---------|-------|
| x² | ✅ Yes | Strictly convex |
| eˣ | ✅ Yes | Strictly convex |
| -ln(x) | ✅ Yes (for x > 0) | Strictly convex |
| \|x\| | ✅ Yes | Convex (not strictly) |
| sin(x) | ❌ No | Oscillates |
| x³ | ❌ No | Changes concavity |

### Properties of Convex Functions

1. **Local minimum = Global minimum**
2. **Sum of convex functions is convex**
3. **Maximum of convex functions is convex**
4. **Composition rules exist**

---

## 5. Convex Optimization Basics

### Standard Form

```
minimize    f(x)
subject to  gᵢ(x) ≤ 0,   i = 1, ..., m
            hⱼ(x) = 0,   j = 1, ..., p

where f and gᵢ are convex, hⱼ are affine
```

### Why Convex Optimization Matters

**Key Property:**
```
For convex problems:
- Any local minimum is a GLOBAL minimum
- No spurious local minima
- Efficient algorithms exist
```

### Examples in ML

| Problem | Convex? | Notes |
|---------|---------|-------|
| Linear Regression (MSE) | ✅ Yes | Closed-form solution |
| Logistic Regression | ✅ Yes | Convex loss |
| Neural Networks | ❌ No | Non-convex, many local minima |
| SVM | ✅ Yes | Convex quadratic program |
| K-Means | ❌ No | Non-convex, sensitive to init |

---

## 6. Jensen's Inequality

### Statement

If f is convex and X is a random variable:

```
f(E[X]) ≤ E[f(X)]
```

**For concave functions, reverse the inequality:**
```
f(E[X]) ≥ E[f(X)]
```

### Examples

**Example 1:** f(x) = x² (convex)
```
(E[X])² ≤ E[X²]

This is why: Var(X) = E[X²] - (E[X])² ≥ 0
```

**Example 2:** f(x) = ln(x) (concave)
```
ln(E[X]) ≥ E[ln(X)]

Used in variational inference, EM algorithm
```

**Example 3:** f(x) = eˣ (convex)
```
e^(E[X]) ≤ E[e^X]

Used in concentration inequalities
```

### Application: ELBO in Variational Inference

```
log p(x) = log ∫ p(x,z) dz
         = log E_q[p(x,z)/q(z)]
         ≥ E_q[log p(x,z) - log q(z)]  (by Jensen's, since log is concave)

This gives the Evidence Lower Bound (ELBO)
```

---

## 7. Lipschitz Continuity

### Definition

A function f is **L-Lipschitz continuous** if:

```
|f(x) - f(y)| ≤ L · \|x - y\|   for all x, y

where L ≥ 0 is the Lipschitz constant
```

**Meaning:** Function cannot change arbitrarily fast; bounded slope.

### Examples

| Function | Lipschitz? | Constant L |
|----------|------------|------------|
| sin(x) | ✅ Yes | L = 1 |
| \|x\| | ✅ Yes | L = 1 |
| x² on [0, 1] | ✅ Yes | L = 2 |
| x² on ℝ | ❌ No | Unbounded derivative |
| eˣ on ℝ | ❌ No | Unbounded derivative |

### Connection to Derivatives

**Theorem:**
```
If f is differentiable and |f'(x)| ≤ L for all x,
then f is L-Lipschitz continuous.
```

### Application: Gradient Descent

**Convergence guarantee:**
```
If ∇f is L-Lipschitz (f has L-Lipschitz gradient),
then gradient descent with step size α ≤ 1/L converges.
```

---

## 8. Uniform Continuity

### Definition

f is **uniformly continuous** if:

```
For every ε > 0, there exists δ > 0 such that:
|x - y| < δ  implies  |f(x) - y| < ε

Key: δ depends only on ε, NOT on x
```

### vs. Regular Continuity

| Continuity | Uniform Continuity |
|------------|-------------------|
| δ can depend on ε AND x | δ depends only on ε |
| Point-wise property | Global property |
| x² is continuous on ℝ | x² is NOT uniformly continuous on ℝ |

### Theorem

```
If f is continuous on a CLOSED, BOUNDED interval,
then f is uniformly continuous on that interval.
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, hessian, exp, sin, cos, ln, series, Matrix

x, y = symbols('x y')

# === Taylor Series ===

# Taylor series of e^x
f = exp(x)
taylor_exp = series(f, x, 0, 6)  # up to x^5
print(f"Taylor series of e^x at 0:\n{taylor_exp}")

# Taylor series of sin(x)
f = sin(x)
taylor_sin = series(f, x, 0, 7)  # up to x^6
print(f"\nTaylor series of sin(x) at 0:\n{taylor_sin}")

# === Multivariate Taylor ===

f = x**2 + 2*x*y + y**2

# Gradient
grad = [diff(f, var) for var in [x, y]]
print(f"\nGradient: {grad}")

# Hessian
H = hessian(f, [x, y])
print(f"\nHessian:\n{H}")

# === Taylor Approximation Visualization ===

def taylor_approximation(func, a, n, x_vals):
    """Compute Taylor polynomial approximation"""
    from sympy import lambdify
    
    # Compute derivatives at a
    derivatives = [func]
    for i in range(n):
        derivatives.append(diff(derivatives[-1], x))
    
    # Evaluate at a
    coeffs = [d.subs(x, a) / np.math.factorial(i) for i, d in enumerate(derivatives)]
    
    # Build polynomial
    taylor_poly = sum(c * (x - a)**i for i, c in enumerate(coeffs))
    
    # Convert to numpy function
    taylor_func = lambdify(x, taylor_poly, 'numpy')
    
    return taylor_func

# Example: Approximate e^x
f = exp(x)
x_vals = np.linspace(-2, 2, 100)
f_np = np.exp

plt.figure(figsize=(12, 5))

# Plot 1: Different order approximations
plt.subplot(1, 2, 1)
plt.plot(x_vals, f_np(x_vals), 'k-', linewidth=3, label='e^x (exact)')

for n in [1, 2, 3, 5]:
    taylor_f = taylor_approximation(f, 0, n, x_vals)
    plt.plot(x_vals, taylor_f(x_vals), '--', linewidth=2, label=f'n={n}')

plt.title('Taylor Approximations of e^x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Approximation error
plt.subplot(1, 2, 2)
for n in [1, 2, 3, 5]:
    taylor_f = taylor_approximation(f, 0, n, x_vals)
    error = np.abs(f_np(x_vals) - taylor_f(x_vals))
    plt.semilogy(x_vals, error, '-', linewidth=2, label=f'n={n}')

plt.title('Approximation Error')
plt.xlabel('x')
plt.ylabel('|Error|')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Newton's Method ===

def newtons_method(f, f_prime, f_double_prime, x0, n_iterations=10):
    """Newton's method for optimization"""
    x = x0
    path = [x]
    
    for _ in range(n_iterations):
        x = x - f_prime(x) / f_double_prime(x)
        path.append(x)
    
    return np.array(path)

# Example: Minimize f(x) = x²
f = lambda x: x**2
f_prime = lambda x: 2*x
f_double_prime = lambda x: 2

path = newtons_method(f, f_prime, f_double_prime, x0=5, n_iterations=5)
print(f"\nNewton's method path: {path}")

# Compare with gradient descent
def gradient_descent(f, f_prime, x0, alpha=0.1, n_iterations=20):
    x = x0
    path = [x]
    
    for _ in range(n_iterations):
        x = x - alpha * f_prime(x)
        path.append(x)
    
    return np.array(path)

gd_path = gradient_descent(f, f_prime, x0=5, alpha=0.1, n_iterations=20)
print(f"Gradient descent path: {gd_path}")

# === Convexity Check ===

def check_convexity(f, x_range=(-10, 10), n_points=100):
    """Numerically check if function is convex"""
    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    
    # Compute second derivative numerically
    h = 1e-5
    f_vals = f(x_vals)
    f_prime = np.gradient(f_vals, x_vals)
    f_double_prime = np.gradient(f_prime, x_vals)
    
    is_convex = np.all(f_double_prime >= -1e-6)  # Allow small numerical error
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, f_vals, 'b-', linewidth=2)
    plt.title(f'f(x): {"Convex" if is_convex else "Not Convex"}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, f_double_prime, 'r-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Second Derivative')
    plt.xlabel('x')
    plt.ylabel("f''(x)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return is_convex

# Test functions
check_convexity(lambda x: x**2, title='x²')
check_convexity(lambda x: np.exp(x), title='e^x')
check_convexity(lambda x: np.sin(x), (-2*np.pi, 2*np.pi), title='sin(x)')

# === Lipschitz Constant Estimation ===

def estimate_lipschitz_constant(f_prime, x_range, n_samples=1000):
    """Estimate Lipschitz constant from derivative"""
    x_vals = np.linspace(x_range[0], x_range[1], n_samples)
    derivative_vals = np.abs(f_prime(x_vals))
    return np.max(derivative_vals)

# Example: f(x) = sin(x), f'(x) = cos(x)
L = estimate_lipschitz_constant(np.cos, (0, 2*np.pi))
print(f"\nEstimated Lipschitz constant for sin(x): L ≈ {L:.4f}")
print(f"Theoretical value: L = 1")

# === Jensen's Inequality Demonstration ===

def demonstrate_jensens():
    """Demonstrate Jensen's inequality"""
    # Convex function: f(x) = x²
    f = lambda x: x**2
    
    # Random variable: uniform on [0, 1]
    samples = np.random.uniform(0, 1, 10000)
    
    # E[X]
    E_X = np.mean(samples)
    
    # f(E[X])
    f_E_X = f(E_X)
    
    # E[f(X)]
    E_f_X = np.mean(f(samples))
    
    print(f"\nJensen's Inequality for f(x) = x²:")
    print(f"f(E[X]) = {f_E_X:.6f}")
    print(f"E[f(X)] = {E_f_X:.6f}")
    print(f"f(E[X]) ≤ E[f(X)]? {f_E_X <= E_f_X} ✓")
    
demonstrate_jensens()
```

---

## 📊 Summary Table

| Concept | Formula | ML Application |
|---------|---------|----------------|
| **Taylor (1st order)** | f(x) ≈ f(a) + ∇f(a)·(x-a) | Linear approximation |
| **Taylor (2nd order)** | f(x) ≈ f(a) + ∇f·Δx + ½ΔxᵀHΔx | Newton's method |
| **Hessian** | Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ | Curvature, optimization |
| **Convex** | f''(x) ≥ 0 or H ⪰ 0 | Global optimization |
| **Jensen's** | f(E[X]) ≤ E[f(X)] | ELBO, variational inference |
| **Lipschitz** | \|f(x)-f(y)\| ≤ L\|x-y\| | Convergence guarantees |

---

## 🎯 ML Applications

| Application | Calculus Concept |
|-------------|-----------------|
| **Backpropagation** | Chain rule, gradients |
| **Newton's Method** | Hessian-based optimization |
| **L-BFGS** | Quasi-Newton, approximate Hessian |
| **Adam Optimizer** | First and second moments |
| **Variational Inference** | Jensen's inequality, ELBO |
| **GAN Training** | Lipschitz constraints (Wasserstein) |
| **BatchNorm** | Taylor expansion analysis |

---

## ❓ Quick Check Questions

1. Why is Taylor series important in ML?
2. What information does the Hessian provide?
3. Why are convex functions desirable in optimization?
4. State Jensen's inequality and give an example.
5. What does Lipschitz continuity guarantee?
6. How does Newton's method differ from gradient descent?

---

## 📝 Answers to Quick Check

1. **Taylor series importance:**
   - Approximates complex functions
   - Foundation for optimization algorithms
   - Analyzes convergence rates

2. **Hessian provides:**
   - Second-order information (curvature)
   - Determines convexity
   - Enables faster optimization (Newton's)

3. **Convex functions desirable because:**
   - Local minimum = global minimum
   - No spurious local minima
   - Efficient algorithms exist

4. **Jensen's inequality:**
   - f(E[X]) ≤ E[f(X)] for convex f
   - Example: (E[X])² ≤ E[X²]
   - Used in variational inference

5. **Lipschitz continuity guarantees:**
   - Bounded rate of change
   - Stability of solutions
   - Convergence of gradient descent

6. **Newton's vs Gradient Descent:**
   - Newton: x - H⁻¹∇f (uses curvature)
   - GD: x - α∇f (uses only gradient)
   - Newton converges faster but is more expensive

---

**Status:** ✅ Complete
**Next:** Practice Problems
