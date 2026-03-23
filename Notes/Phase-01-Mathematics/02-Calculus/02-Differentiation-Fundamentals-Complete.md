# 1.2.2 Differentiation Fundamentals

## 🎯 Quick Overview
- **Derivative**: Instantaneous rate of change
- **Geometric meaning**: Slope of tangent line
- **Foundation for**: Gradient descent, backpropagation, optimization

---

## 1. Derivative as a Limit

### Formal Definition

The **derivative** of f(x) at point x is:

```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

**Alternative notation:**
```
dy/dx = lim(Δx→0) Δy/Δx
```

### Geometric Interpretation

```
Secant line: connects (x, f(x)) and (x+h, f(x+h))
  Slope = [f(x+h) - f(x)] / h

Tangent line: limit of secant as h→0
  Slope = f'(x)
```

**Visual:**
```
        /
       /  ← Curve y = f(x)
      /|
     / |
    /  | Δy = f(x+h) - f(x)
   /   |
  /____|
   Δx = h

Slope of secant = Δy/Δx
Slope of tangent = lim(Δx→0) Δy/Δx
```

---

## 2. Derivative Notation

| Notation | Context | Read as |
|----------|---------|---------|
| **f'(x)** | Lagrange | "f prime of x" |
| **dy/dx** | Leibniz | "dy dx" or "derivative of y wrt x" |
| **D_x f** | Euler | "D sub x of f" |
| **ḟ(x)** | Newton | "f dot" (time derivatives) |
| **∂f/∂x** | Partial | "partial f partial x" (multivariable) |

**Higher-order derivatives:**
```
f'(x)    = first derivative
f''(x)   = second derivative
f'''(x)  = third derivative
f⁽ⁿ⁾(x)  = nth derivative

d²y/dx²  = second derivative (Leibniz)
```

---

## 3. Differentiation Rules

### Basic Rules

| Rule | Formula | Example |
|------|---------|---------|
| **Constant** | d/dx(c) = 0 | d/dx(5) = 0 |
| **Identity** | d/dx(x) = 1 | d/dx(x) = 1 |
| **Power** | d/dx(xⁿ) = nxⁿ⁻¹ | d/dx(x³) = 3x² |
| **Constant Multiple** | d/dx(cf) = c·f'(x) | d/dx(3x²) = 6x |
| **Sum/Difference** | d/dx(f±g) = f'±g' | d/dx(x²+x) = 2x+1 |

### Product Rule

```
d/dx[f(x)·g(x)] = f'(x)g(x) + f(x)g'(x)

Mnemonic: "derivative of first times second, plus first times derivative of second"
```

**Example:**
```
f(x) = x²·sin(x)

f'(x) = (2x)·sin(x) + x²·cos(x)
```

### Quotient Rule

```
d/dx[f(x)/g(x)] = [f'(x)g(x) - f(x)g'(x)] / [g(x)]²

Mnemonic: "low d-high minus high d-low, over low squared"
```

**Example:**
```
        x² + 1
f(x) = ───────
        x - 1

        (2x)(x-1) - (x²+1)(1)    2x² - 2x - x² - 1    x² - 2x - 1
f'(x) = ───────────────────── = ─────────────────── = ───────────
              (x-1)²                   (x-1)²            (x-1)²
```

### Chain Rule ⭐ CRITICAL

```
d/dx[f(g(x))] = f'(g(x)) · g'(x)

Or in Leibniz notation:
dy/dx = dy/du · du/dx  where y = f(u), u = g(x)
```

**Example 1:**
```
f(x) = sin(x²)

Let u = x², then f = sin(u)

f'(x) = cos(u) · 2x = cos(x²) · 2x = 2x·cos(x²)
```

**Example 2:**
```
f(x) = e^(3x+1)

Let u = 3x+1, then f = e^u

f'(x) = e^u · 3 = 3e^(3x+1)
```

**Example 3 (Multiple compositions):**
```
f(x) = sin²(cos(e^x))

f'(x) = 2sin(cos(e^x)) · cos(cos(e^x)) · (-sin(e^x)) · e^x
```

---

## 4. Derivatives of Elementary Functions

### Exponential and Logarithmic

| Function | Derivative | Notes |
|----------|------------|-------|
| eˣ | eˣ | Only function equal to its derivative! |
| aˣ | aˣ·ln(a) | |
| ln(x) | 1/x | x > 0 |
| logₐ(x) | 1/(x·ln(a)) | |

### Trigonometric

| Function | Derivative | Domain |
|----------|------------|--------|
| sin(x) | cos(x) | ℝ |
| cos(x) | -sin(x) | ℝ |
| tan(x) | sec²(x) | x ≠ π/2 + nπ |
| sec(x) | sec(x)tan(x) | x ≠ π/2 + nπ |
| csc(x) | -csc(x)cot(x) | x ≠ nπ |
| cot(x) | -csc²(x) | x ≠ nπ |

**Mnemonic:** Derivatives of "co" functions are negative!

### Inverse Trigonometric

| Function | Derivative | Domain |
|----------|------------|--------|
| arcsin(x) | 1/√(1-x²) | \|x\| < 1 |
| arccos(x) | -1/√(1-x²) | \|x\| < 1 |
| arctan(x) | 1/(1+x²) | ℝ |
| arcsec(x) | 1/(\|x\|√(x²-1)) | \|x\| > 1 |

---

## 5. Implicit Differentiation

### When to Use

When y cannot be easily solved in terms of x:
```
x² + y² = 25  (circle)
xy + sin(y) = x²
```

### Method

**Example:** Find dy/dx for x² + y² = 25

```
Step 1: Differentiate both sides wrt x
d/dx(x²) + d/dx(y²) = d/dx(25)

Step 2: Apply chain rule to y terms
2x + 2y·(dy/dx) = 0

Step 3: Solve for dy/dx
2y·(dy/dx) = -2x
dy/dx = -x/y
```

**Another Example:**
```
xy = 1

Differentiate:
x·(dy/dx) + y·1 = 0

Solve:
dy/dx = -y/x = -y/(1/y) = -y²
```

---

## 6. Logarithmic Differentiation

### When to Use

- Functions with variables in both base and exponent: y = f(x)^g(x)
- Complicated products/quotients
- Simplifies differentiation using log properties

### Method

**Example:** y = xˣ (x > 0)

```
Step 1: Take ln of both sides
ln(y) = ln(xˣ) = x·ln(x)

Step 2: Differentiate implicitly
(1/y)·y' = ln(x) + x·(1/x) = ln(x) + 1

Step 3: Solve for y'
y' = y·(ln(x) + 1) = xˣ·(ln(x) + 1)
```

**Another Example:**
```
        (x+1)²·√(x-1)
y = ─────────────────
        (x+2)³

Take ln:
ln(y) = 2ln(x+1) + (1/2)ln(x-1) - 3ln(x+2)

Differentiate:
y'/y = 2/(x+1) + 1/(2(x-1)) - 3/(x+2)

y' = y · [2/(x+1) + 1/(2(x-1)) - 3/(x+2)]
```

---

## 7. Higher-Order Derivatives

### Definition

```
f'(x)   = first derivative (rate of change)
f''(x)  = second derivative (rate of change of rate)
f'''(x) = third derivative
f⁽ⁿ⁾(x) = nth derivative
```

### Physical Interpretation

For position s(t):
```
s'(t)  = velocity (rate of change of position)
s''(t) = acceleration (rate of change of velocity)
s'''(t) = jerk (rate of change of acceleration)
```

### Examples

**Example 1:** f(x) = x⁴
```
f'(x)  = 4x³
f''(x) = 12x²
f'''(x) = 24x
f⁽⁴⁾(x) = 24
f⁽⁵⁾(x) = 0
```

**Example 2:** f(x) = eˣ
```
f'(x) = f''(x) = f'''(x) = ... = eˣ
```

**Example 3:** f(x) = sin(x)
```
f'(x)  = cos(x)
f''(x) = -sin(x)
f'''(x) = -cos(x)
f⁽⁴⁾(x) = sin(x)  (back to original!)

Pattern repeats every 4 derivatives
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin, cos, exp, ln, simplify, latex

x = symbols('x')

# === Symbolic Differentiation ===

# Basic derivatives
f = x**3 + 2*x**2 - 5*x + 1
f_prime = diff(f, x)
print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")

# Product rule
f = x**2 * sin(x)
f_prime = diff(f, x)
print(f"\nd/dx[x²·sin(x)] = {f_prime}")

# Quotient rule
f = (x**2 + 1) / (x - 1)
f_prime = diff(f, x)
print(f"\nd/dx[(x²+1)/(x-1)] = {simplify(f_prime)}")

# Chain rule
f = sin(x**2)
f_prime = diff(f, x)
print(f"\nd/dx[sin(x²)] = {f_prime}")

# Higher-order derivatives
f = x**5 - 3*x**3 + 2*x
derivatives = [f]
for i in range(5):
    derivatives.append(diff(derivatives[-1], x))

print("\nHigher-order derivatives of x⁵ - 3x³ + 2x:")
for i, d in enumerate(derivatives):
    print(f"  f⁽{i}⁾(x) = {d}")

# === Numerical Differentiation ===

def numerical_derivative(f, x, h=1e-5):
    """Compute derivative using limit definition"""
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h=1e-5):
    """More accurate: central difference"""
    return (f(x + h) - f(x - h)) / (2*h)

# Example: derivative of x² at x=3
f = lambda x: x**2
x_val = 3

forward = numerical_derivative(f, x_val)
central = central_difference(f, x_val)
exact = 2 * x_val  # analytical: d/dx(x²) = 2x

print(f"\nDerivative of x² at x=3:")
print(f"  Forward difference: {forward:.10f}")
print(f"  Central difference: {central:.10f}")
print(f"  Exact (2x): {exact}")

# === Visualization ===

# Plot function and tangent line
def plot_tangent(f, f_prime, x0, x_range=(-3, 3)):
    """Plot function with tangent line at x0"""
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = f(x_vals)
    
    # Tangent line: y = f(x0) + f'(x0)(x - x0)
    tangent_y = f(x0) + f_prime(x0) * (x_vals - x0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    plt.plot(x_vals, tangent_y, 'r--', linewidth=2, label=f"Tangent at x={x0}")
    plt.plot(x0, f(x0), 'go', markersize=10, label=f'Point ({x0}, {f(x0):.2f})')
    plt.axvline(x=x0, color='g', linestyle=':', alpha=0.5)
    plt.title(f'Function and Tangent Line at x = {x0}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: f(x) = x²
f = lambda x: x**2
f_prime = lambda x: 2*x
plot_tangent(f, f_prime, x0=1.5, x_range=(-3, 3))

# Example: f(x) = sin(x)
f = lambda x: np.sin(x)
f_prime = lambda x: np.cos(x)
plot_tangent(f, f_prime, x0=np.pi/4, x_range=(-np.pi, np.pi))

# === Derivative Table Generator ===

def create_derivative_table():
    """Create a table of common derivatives"""
    functions = [
        ('x^n', lambda x, n=3: x**3, lambda x, n=3: 3*x**2, 'Power rule'),
        ('e^x', np.exp, np.exp, 'Exponential'),
        ('ln(x)', np.log, lambda x: 1/x, 'Logarithm'),
        ('sin(x)', np.sin, np.cos, 'Trig'),
        ('cos(x)', np.cos, lambda x: -np.sin(x), 'Trig'),
    ]
    
    print("\nDerivative Table at x=1:")
    print("-" * 60)
    print(f"{'Function':<15} {'f(1)':<15} {'f\'(1)':<15} {'Rule'}")
    print("-" * 60)
    
    for name, f, f_prime, rule in functions:
        try:
            val = f(1)
            prime_val = f_prime(1)
            print(f"{name:<15} {val:<15.6f} {prime_val:<15.6f} {rule}")
        except:
            print(f"{name:<15} {'N/A':<15} {'N/A':<15} {rule}")
    
create_derivative_table()
```

---

## 📊 Summary Table

| Rule | Formula | When to Use |
|------|---------|-------------|
| **Power** | d/dx(xⁿ) = nxⁿ⁻¹ | Polynomial terms |
| **Product** | (fg)' = f'g + fg' | Multiplication |
| **Quotient** | (f/g)' = (f'g - fg')/g² | Division |
| **Chain** | d/dx(f(g(x))) = f'(g(x))·g'(x) | Composition |
| **Implicit** | Differentiate both sides | y not isolated |
| **Log** | Take ln, then differentiate | x^x, complex products |

---

## 🎯 ML Applications

| Application | Derivative Concept |
|-------------|-------------------|
| **Gradient Descent** | d(loss)/d(weights) |
| **Backpropagation** | Chain rule extensively |
| **Activation Functions** | d/dx(ReLU), d/dx(sigmoid), etc. |
| **Optimization** | Finding minima via f'(x)=0 |
| **Taylor Expansion** | Higher-order derivatives |

**Backpropagation Example:**
```
Neural network: output = f(g(h(x)))

∂output/∂x = f'(g(h(x))) · g'(h(x)) · h'(x)

This is the chain rule!
```

---

## ❓ Quick Check Questions

1. What is the geometric meaning of the derivative?
2. When would you use the chain rule vs product rule?
3. Why is d/dx(eˣ) = eˣ special?
4. How do you differentiate y = xˣ?
5. What does the second derivative tell you?
6. When is implicit differentiation necessary?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Geometric meaning:**
   - Slope of the tangent line at a point
   - Instantaneous rate of change

2. **Chain vs Product:**
   - Chain rule: composition f(g(x))
   - Product rule: multiplication f(x)·g(x)

3. **eˣ is special because:**
   - It equals its own derivative
   - Appears naturally in growth/decay
   - Base of natural logarithm

4. **Differentiating y = xˣ:**
   - Use logarithmic differentiation
   - ln(y) = x·ln(x)
   - y' = xˣ(ln(x) + 1)

5. **Second derivative tells:**
   - Concavity (f'' > 0: concave up)
   - Acceleration (in physics)
   - Curvature of graph

6. **Implicit differentiation needed when:**
   - y cannot be easily solved for x
   - Equation defines y implicitly (e.g., circle)

</details>
---

**Status:** ✅ Complete
**Next:** Applications of Derivatives
