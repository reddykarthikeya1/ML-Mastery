# 1.2.4 Partial Derivatives

## 🎯 Quick Overview
- **Partial derivative**: Derivative with respect to one variable, holding others constant
- **Notation**: ∂f/∂x, f_x
- **Foundation for**: Gradient, Hessian, multivariable optimization, backpropagation

---

## 1. Functions of Several Variables

### Definition

A function of **two variables** assigns a unique output z to each pair (x, y):
```
z = f(x, y)
```

**Domain:** Set of all valid (x, y) pairs
**Range:** Set of all possible z values

### Examples

| Function | Domain | Visualization |
|----------|--------|---------------|
| f(x, y) = x² + y² | ℝ² | Paraboloid (bowl) |
| f(x, y) = √(x² + y²) | ℝ² | Cone |
| f(x, y) = xy | ℝ² | Saddle |
| f(x, y) = 1/(x² + y²) | ℝ² \ {(0,0)} | Hill with hole |
| f(x, y, z) = x² + y² + z² | ℝ³ | 4D hypersphere |

### Visualizing Functions of Two Variables

**Method 1: Surface Plot (3D)**
```
z = f(x, y) plotted in 3D space
```

**Method 2: Contour Plot (2D)**
```
Level curves: f(x, y) = c (constant)
Each curve shows where function has same value
```

**Method 3: Heat Map**
```
Color represents z value at each (x, y)
```

---

## 2. Limits and Continuity in Multiple Variables

### Limit Definition

```
lim((x,y)→(a,b)) f(x, y) = L

means: f(x, y) approaches L as (x, y) approaches (a, b)
       from ANY direction in the plane
```

**Key Difference from 1D:**
- In 1D: approach from left OR right
- In 2D: approach from INFINITELY many directions!

### When Limit Does NOT Exist

If f approaches different values along different paths, the limit does not exist.

**Example:**
```
        xy
f(x, y) = ───────
        x² + y²

Path 1: Along x-axis (y = 0)
lim((x,0)→(0,0)) f(x, 0) = lim(x→0) 0/x² = 0

Path 2: Along y-axis (x = 0)
lim((0,y)→(0,0)) f(0, y) = lim(y→0) 0/y² = 0

Path 3: Along line y = x
lim((x,x)→(0,0)) f(x, x) = lim(x→0) x²/(2x²) = 1/2

Different paths give different limits!
Therefore: lim((x,y)→(0,0)) f(x, y) DOES NOT EXIST
```

### Continuity

f(x, y) is **continuous** at (a, b) if:
```
lim((x,y)→(a,b)) f(x, y) = f(a, b)
```

**Polynomials and rational functions are continuous on their domains.**

---

## 3. Partial Derivatives

### Definition

The **partial derivative** of f(x, y) with respect to x:

```
∂f/∂x = lim(h→0) [f(x+h, y) - f(x, y)] / h

Treat y as CONSTANT, differentiate with respect to x
```

### Notation

| Notation | Read as | Context |
|----------|---------|---------|
| ∂f/∂x | "partial f partial x" | Leibniz |
| f_x | "f sub x" | Subscript |
| D₁f | "D sub 1 of f" | Operator |
| z_x | "z sub x" | When z = f(x,y) |

### Computation Rules

**Same rules as single-variable calculus:**
- Power rule
- Product rule
- Quotient rule
- Chain rule

**Key:** Treat other variables as CONSTANTS!

### Examples

**Example 1:** f(x, y) = x³y² + 2xy - y³

```
∂f/∂x = 3x²y² + 2y      (treat y as constant)
∂f/∂y = 2x³y + 2x - 3y² (treat x as constant)
```

**Example 2:** f(x, y) = e^(xy)

```
∂f/∂x = y·e^(xy)        (chain rule, y is constant)
∂f/∂y = x·e^(xy)        (chain rule, x is constant)
```

**Example 3:** f(x, y) = sin(x² + y²)

```
∂f/∂x = cos(x² + y²) · 2x = 2x·cos(x² + y²)
∂f/∂y = cos(x² + y²) · 2y = 2y·cos(x² + y²)
```

**Example 4:** f(x, y, z) = x²y + yz² + xz

```
∂f/∂x = 2xy + z
∂f/∂y = x² + z²
∂f/∂z = 2yz + x
```

---

## 4. Higher-Order Partial Derivatives

### Second-Order Partials

For f(x, y):

| Notation | Meaning | Compute |
|----------|---------|---------|
| f_xx | ∂²f/∂x² | ∂/∂x(∂f/∂x) |
| f_yy | ∂²f/∂y² | ∂/∂y(∂f/∂y) |
| f_xy | ∂²f/∂y∂x | ∂/∂y(∂f/∂x) |
| f_yx | ∂²f/∂x∂y | ∂/∂x(∂f/∂y) |

**Note:** f_xy means differentiate x FIRST, then y.

### Clairaut's Theorem

**If mixed partials are continuous, then:**
```
f_xy = f_yx
```

**Example:**
```
f(x, y) = x³y² + 2xy

f_x = 3x²y² + 2y
f_y = 2x³y + 2x

f_xx = 6xy²
f_yy = 2x³

f_xy = ∂/∂y(3x²y² + 2y) = 6x²y + 2
f_yx = ∂/∂x(2x³y + 2x) = 6x²y + 2  ✓

f_xy = f_yx as expected!
```

### Example with Three Variables

```
f(x, y, z) = x²y + yz² + xz

First derivatives:
f_x = 2xy + z
f_y = x² + z²
f_z = 2yz + x

Second derivatives:
f_xx = 2y
f_yy = 0
f_zz = 2y

f_xy = 2x
f_xz = 1
f_yz = 2z

(By Clairaut: f_yx = f_xy = 2x, etc.)
```

---

## 5. Tangent Planes and Linear Approximation

### Tangent Plane

For surface z = f(x, y), the **tangent plane** at (a, b, f(a,b)):

```
z = f(a, b) + f_x(a, b)(x - a) + f_y(a, b)(y - b)
```

**Analogy to 1D:**
```
1D: y = f(a) + f'(a)(x - a)  (tangent line)
2D: z = f(a,b) + f_x(a,b)(x-a) + f_y(a,b)(y-b)  (tangent plane)
```

### Example

```
Find tangent plane to z = x² + y² at (1, 2, 5):

f(x, y) = x² + y²
f_x = 2x  →  f_x(1, 2) = 2
f_y = 2y  →  f_y(1, 2) = 4

Tangent plane:
z = 5 + 2(x - 1) + 4(y - 2)
z = 5 + 2x - 2 + 4y - 8
z = 2x + 4y - 5
```

### Linear Approximation

**Near (a, b):**
```
f(x, y) ≈ f(a, b) + f_x(a, b)(x - a) + f_y(a, b)(y - b)
```

**Error estimate:**
```
|Error| ≤ M · √[(x-a)² + (y-b)²]

where M bounds the second derivatives
```

### Example: Estimation

```
Estimate √(3.98² + 3.01²) using linear approximation:

f(x, y) = √(x² + y²)
Choose (a, b) = (4, 3) since √(16+9) = 5 is easy

f_x = x/√(x²+y²)  →  f_x(4,3) = 4/5 = 0.8
f_y = y/√(x²+y²)  →  f_y(4,3) = 3/5 = 0.6

Linear approximation:
f(3.98, 3.01) ≈ 5 + 0.8(3.98-4) + 0.6(3.01-3)
              = 5 + 0.8(-0.02) + 0.6(0.01)
              = 5 - 0.016 + 0.006
              = 4.99

Actual: √(3.98² + 3.01²) ≈ 4.9901... ✓
```

---

## 6. Differentiability in Multiple Variables

### Definition

f(x, y) is **differentiable** at (a, b) if:
```
f(x, y) = f(a, b) + f_x(a, b)(x-a) + f_y(a, b)(y-b) + ε(x, y)

where ε(x,y)/√[(x-a)²+(y-b)²] → 0 as (x,y)→(a,b)
```

**Meaning:** Function is well-approximated by tangent plane.

### Theorem

**If:**
- f_x and f_y exist AND are continuous near (a, b)

**Then:**
- f is differentiable at (a, b)

### Important Distinction

| Property | Implies |
|----------|---------|
| Differentiable | → Continuous |
| Differentiable | → Partial derivatives exist |
| Partial derivatives exist | ↛ Differentiable |
| Continuous | ↛ Differentiable |

**Example where partials exist but not differentiable:**
```
        { xy/(x² + y²),  (x,y) ≠ (0,0)
f(x, y) = {
        { 0,             (x,y) = (0,0)

f_x(0,0) = 0 and f_y(0,0) = 0 (can verify)
BUT f is not continuous at (0,0), so not differentiable
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, exp, sin, cos, ln, sqrt
from mpl_toolkits.mplot3d import Axes3D

x, y, z = symbols('x y z')

# === Symbolic Partial Derivatives ===

# Example 1: Polynomial
f = x**3 * y**2 + 2*x*y - y**3

f_x = diff(f, x)
f_y = diff(f, y)

print(f"f(x, y) = {f}")
print(f"∂f/∂x = {f_x}")
print(f"∂f/∂y = {f_y}")

# Second derivatives
f_xx = diff(f_x, x)
f_yy = diff(f_y, y)
f_xy = diff(f_x, y)
f_yx = diff(f_y, x)

print(f"\nSecond derivatives:")
print(f"∂²f/∂x² = {f_xx}")
print(f"∂²f/∂y² = {f_yy}")
print(f"∂²f/∂y∂x = {f_xy}")
print(f"∂²f/∂x∂y = {f_yx}")
print(f"f_xy = f_yx? {f_xy == f_yx}")

# Example 2: Exponential
f2 = exp(x * y)
print(f"\nf(x,y) = e^(xy)")
print(f"∂f/∂x = {diff(f2, x)}")
print(f"∂f/∂y = {diff(f2, y)}")

# === Numerical Partial Derivatives ===

def numerical_partial_x(f, x, y, h=1e-5):
    """Compute ∂f/∂x numerically"""
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def numerical_partial_y(f, x, y, h=1e-5):
    """Compute ∂f/∂y numerically"""
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

# Test
f = lambda x, y: x**2 * y + y**2
x0, y0 = 2, 3

numerical_fx = numerical_partial_x(f, x0, y0)
numerical_fy = numerical_partial_y(f, x0, y0)

# Analytical
analytical_fx = 2*x0*y0  # 2*2*3 = 12
analytical_fy = x0**2 + 2*y0  # 4 + 6 = 10

print(f"\nAt ({x0}, {y0}):")
print(f"Numerical ∂f/∂x = {numerical_fx:.6f}, Analytical = {analytical_fx}")
print(f"Numerical ∂f/∂y = {numerical_fy:.6f}, Analytical = {analytical_fy}")

# === 3D Surface Plot ===

def plot_surface(f, x_range, y_range, title="3D Surface"):
    """Plot 3D surface"""
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_title(f'{title} - Surface')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax2.set_title(f'{title} - Contour Plot')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(contour, ax=ax2, label='z')
    
    plt.tight_layout()
    plt.show()

# Example surfaces
plot_surface(lambda x, y: x**2 + y**2, (-3, 3), (-3, 3), "z = x² + y² (Paraboloid)")
plot_surface(lambda x, y: x*y, (-3, 3), (-3, 3), "z = xy (Saddle)")
plot_surface(lambda x, y: np.sqrt(x**2 + y**2), (-3, 3), (-3, 3), "z = √(x² + y²) (Cone)")

# === Tangent Plane Visualization ===

def plot_tangent_plane(f, f_x, f_y, point, x_range, y_range):
    """Plot surface with tangent plane"""
    a, b = point
    c = f(a, b)
    
    # Partial derivatives at point
    fx_val = f_x(a, b)
    fy_val = f_y(a, b)
    
    # Tangent plane function
    def tangent_plane(x, y):
        return c + fx_val * (x - a) + fy_val * (y - b)
    
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z_surface = f(X, Y)
    Z_tangent = tangent_plane(X, Y)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface
    ax.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=0.6, label='Surface')
    
    # Tangent plane (smaller region around point)
    x_tan = np.linspace(a - 1, a + 1, 20)
    y_tan = np.linspace(b - 1, b + 1, 20)
    X_tan, Y_tan = np.meshgrid(x_tan, y_tan)
    Z_tan = tangent_plane(X_tan, Y_tan)
    ax.plot_surface(X_tan, Y_tan, Z_tan, color='red', alpha=0.5, label='Tangent Plane')
    
    # Point of tangency
    ax.scatter(a, b, c, color='black', s=100, label=f'Point ({a}, {b}, {c})')
    
    ax.set_title(f'Surface and Tangent Plane at ({a}, {b})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    
    plt.show()
    
    # Print tangent plane equation
    print(f"Tangent plane at ({a}, {b}):")
    print(f"z = {c} + {fx_val}(x - {a}) + {fy_val}(y - {b})")
    print(f"z = {fx_val}x + {fy_val}y + {c - fx_val*a - fy_val*b}")

# Example
f = lambda x, y: x**2 + y**2
f_x = lambda x, y: 2*x
f_y = lambda x, y: 2*y

plot_tangent_plane(f, f_x, f_y, point=(1, 2), x_range=(-3, 3), y_range=(-3, 3))

# === Gradient Field with Contours ===

def plot_gradient_field(f, partial_x, partial_y, x_range, y_range, density=15):
    """Plot contours with gradient vectors"""
    x_vals = np.linspace(x_range[0], x_range[1], density)
    y_vals = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)
    
    # Compute gradient
    U = partial_x(X, Y)
    V = partial_y(X, Y)
    
    # Normalize for visualization
    N = np.sqrt(U**2 + V**2)
    U_norm = U / np.where(N > 0, N, 1)
    V_norm = V / np.where(N > 0, N, 1)
    
    plt.figure(figsize=(10, 8))
    
    # Contours
    contour = plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    
    # Gradient vectors
    plt.quiver(X, Y, U_norm, V_norm, color='white', alpha=0.8, scale=25, width=0.003)
    
    plt.title('Function Contours with Gradient Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(contour, label='f(x,y)')
    plt.grid(True, alpha=0.3)
    plt.show()

# Example
f = lambda x, y: x**2 - y**2  # Saddle
partial_x = lambda x, y: 2*x
partial_y = lambda x, y: -2*y

plot_gradient_field(f, partial_x, partial_y, (-3, 3), (-3, 3))

# === Linear Approximation Error ===

def linear_approximation_demo(f, f_x, f_y, point, test_points):
    """Demonstrate linear approximation accuracy"""
    a, b = point
    c = f(a, b)
    fx_val = f_x(a, b)
    fy_val = f_y(a, b)
    
    def linear_approx(x, y):
        return c + fx_val * (x - a) + fy_val * (y - b)
    
    print(f"Linear approximation at ({a}, {b}):")
    print(f"L(x,y) = {c} + {fx_val}(x-{a}) + {fy_val}(y-{b})")
    print(f"\n{'Point':<20} {'Actual':<15} {'Approx':<15} {'Error':<15}")
    print("-" * 65)
    
    for pt in test_points:
        x, y = pt
        actual = f(x, y)
        approx = linear_approx(x, y)
        error = abs(actual - approx)
        print(f"({x}, {y})".ljust(20) + f"{actual:.6f}".ljust(15) + 
              f"{approx:.6f}".ljust(15) + f"{error:.6f}")

# Example
f = lambda x, y: np.exp(x + y)
f_x = lambda x, y: np.exp(x + y)
f_y = lambda x, y: np.exp(x + y)

test_pts = [(0.1, 0.1), (0.2, -0.1), (-0.1, 0.2), (0.5, 0.5)]
linear_approximation_demo(f, f_x, f_y, point=(0, 0), test_points=test_pts)
```

---

## 📊 Summary Table

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Partial ∂f/∂x** | lim(h→0) [f(x+h,y)-f(x,y)]/h | Rate of change in x-direction |
| **Partial ∂f/∂y** | lim(h→0) [f(x,y+h)-f(x,y)]/h | Rate of change in y-direction |
| **f_xy** | ∂/∂y(∂f/∂x) | Mixed partial |
| **Tangent Plane** | z = f(a,b) + f_x(a,b)(x-a) + f_y(a,b)(y-b) | Linear approximation |
| **Linear Approx** | L(x,y) = f(a,b) + f_x(a,b)(x-a) + f_y(a,b)(y-b) | Approximation near (a,b) |

---

## 🎯 ML Applications

| Application | Partial Derivative Concept |
|-------------|---------------------------|
| **Gradient Descent** | ∇L = (∂L/∂w₁, ∂L/∂w₂, ...) |
| **Backpropagation** | Chain rule for partial derivatives |
| **Neural Networks** | ∂Loss/∂weight for each weight |
| **Hessian Matrix** | All second partial derivatives |
| **Feature Importance** | ∂Output/∂Featureᵢ |

**Backpropagation Example:**
```
Neural network: L = Loss(y_pred, y_true)
                y_pred = f(W₂ · g(W₁ · x))

Chain rule:
∂L/∂W₂ = ∂L/∂y_pred · ∂y_pred/∂W₂
∂L/∂W₁ = ∂L/∂y_pred · ∂y_pred/∂y_hidden · ∂y_hidden/∂W₁

This is computing partial derivatives!
```

---

## ❓ Quick Check Questions

1. What does ∂f/∂x mean geometrically?
2. How do you compute a partial derivative?
3. State Clairaut's theorem.
4. What is the tangent plane equation?
5. When does a limit NOT exist in multivariable calculus?
6. What's the relationship between differentiability and continuity?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **∂f/∂x geometric meaning:**
   - Slope of surface in x-direction
   - Rate of change of f as x varies (y fixed)

2. **Computing partial derivatives:**
   - Treat other variables as constants
   - Apply standard differentiation rules

3. **Clairaut's theorem:**
   - If mixed partials are continuous
   - Then f_xy = f_yx

4. **Tangent plane equation:**
   - z = f(a,b) + f_x(a,b)(x-a) + f_y(a,b)(y-b)

5. **Limit doesn't exist when:**
   - Different paths give different limits
   - Function oscillates infinitely
   - Function goes to infinity

6. **Differentiability vs Continuity:**
   - Differentiable → Continuous
   - Continuous ↛ Differentiable
   - Partials exist ↛ Differentiable

</details>
---

**Status:** ✅ Complete
**Next:** Chain Rule for Multivariable Functions
