# 1.2.6 Chain Rule for Multivariable Functions

## 🎯 Quick Overview
- **Chain Rule**: Differentiate composite functions
- **Tree diagrams**: Visual way to track dependencies
- **Foundation for**: Backpropagation, automatic differentiation

---

## 1. Chain Rule for Composite Functions

### Single Variable Review

For h(x) = f(g(x)):
```
h'(x) = f'(g(x)) · g'(x)
```

### Multivariable Case 1: One Intermediate Variable

**Scenario:** z = f(x, y) where x = x(t), y = y(t)

**Chain Rule:**
```
dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)

Or: dz/dt = f_x · dx/dt + f_y · dy/dt
```

**Tree Diagram:**
```
        z
       / \
      x   y
      |   |
      t   t

dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
```

### Example 1

```
z = x² + y², x = t², y = t³

Find dz/dt:

Method 1 (Chain Rule):
∂z/∂x = 2x, ∂z/∂y = 2y
dx/dt = 2t, dy/dt = 3t²

dz/dt = 2x·2t + 2y·3t²
      = 2(t²)·2t + 2(t³)·3t²
      = 4t³ + 6t⁵

Method 2 (Direct substitution):
z = (t²)² + (t³)² = t⁴ + t⁶
dz/dt = 4t³ + 6t⁵  ✓
```

---

### Multivariable Case 2: Two Intermediate Variables

**Scenario:** z = f(x, y) where x = x(s, t), y = y(s, t)

**Chain Rule:**
```
∂z/∂s = (∂f/∂x)(∂x/∂s) + (∂f/∂y)(∂y/∂s)
∂z/∂t = (∂f/∂x)(∂x/∂t) + (∂f/∂y)(∂y/∂t)
```

**Tree Diagram:**
```
        z
       / \
      x   y
     / \ / \
    s   t   s   t

∂z/∂s = (∂z/∂x)(∂x/∂s) + (∂z/∂y)(∂y/∂s)
∂z/∂t = (∂z/∂x)(∂x/∂t) + (∂z/∂y)(∂y/∂t)
```

### Example 2

```
z = eˣ·cos(y), x = s² - t², y = 2st

Find ∂z/∂s and ∂z/∂t:

∂z/∂x = eˣ·cos(y)
∂z/∂y = -eˣ·sin(y)

∂x/∂s = 2s, ∂x/∂t = -2t
∂y/∂s = 2t, ∂y/∂t = 2s

∂z/∂s = eˣ·cos(y)·2s + (-eˣ·sin(y))·2t
       = 2s·eˣ·cos(y) - 2t·eˣ·sin(y)
       = 2eˣ[s·cos(y) - t·sin(y)]

∂z/∂t = eˣ·cos(y)·(-2t) + (-eˣ·sin(y))·2s
       = -2t·eˣ·cos(y) - 2s·eˣ·sin(y)
       = -2eˣ[t·cos(y) + s·sin(y)]
```

---

### Multivariable Case 3: Three Variables

**Scenario:** w = f(x, y, z) where x = x(t), y = y(t), z = z(t)

**Chain Rule:**
```
dw/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt) + (∂f/∂z)(dz/dt)
```

**Tree Diagram:**
```
        w
      / | \
     x  y  z
     |  |  |
     t  t  t

dw/dt = (∂w/∂x)(dx/dt) + (∂w/∂y)(dy/dt) + (∂w/∂z)(dz/dt)
```

### Example 3

```
w = x²y + yz², x = eᵗ, y = t², z = sin(t)

Find dw/dt:

∂w/∂x = 2xy
∂w/∂y = x² + z²
∂w/∂z = 2yz

dx/dt = eᵗ
dy/dt = 2t
dz/dt = cos(t)

dw/dt = 2xy·eᵗ + (x² + z²)·2t + 2yz·cos(t)
      = 2(eᵗ)(t²)·eᵗ + ((eᵗ)² + sin²(t))·2t + 2(t²)sin(t)·cos(t)
      = 2t²e²ᵗ + 2t(e²ᵗ + sin²(t)) + t²sin(2t)
```

---

## 2. Tree Diagrams for Chain Rule

### How to Draw Tree Diagrams

**Step 1:** Start with dependent variable at top
**Step 2:** Branch to all intermediate variables
**Step 3:** Branch from each to independent variables
**Step 4:** Label each branch with appropriate derivative
**Step 5:** Multiply along paths, add results

### General Pattern

For z = f(x₁, x₂, ..., xₙ) where each xᵢ = xᵢ(t₁, t₂, ..., tₘ):

```
∂z/∂tⱼ = Σᵢ (∂f/∂xᵢ)(∂xᵢ/∂tⱼ)
```

### Example with Complex Dependencies

```
w = f(x, y, z)
x = x(u, v)
y = y(u, v, w)  # Note: w appears on both sides!
z = z(u, v)

Find ∂w/∂u:

Tree:
        w
      / | \
     x  y  z
    / \ / \ / \
   u  v u  v u  v

∂w/∂u = (∂w/∂x)(∂x/∂u) + (∂w/∂y)(∂y/∂u) + (∂w/∂z)(∂z/∂u)
```

---

## 3. Implicit Differentiation in Multiple Variables

### Case 1: F(x, y) = 0

If F(x, y) = 0 defines y implicitly as a function of x:

```
dy/dx = -F_x / F_y = -(∂F/∂x) / (∂F/∂y)
```

**Derivation:**
```
F(x, y) = 0

Differentiate both sides wrt x:
(∂F/∂x)(dx/dx) + (∂F/∂y)(dy/dx) = 0

F_x · 1 + F_y · (dy/dx) = 0

dy/dx = -F_x / F_y
```

### Example 4

```
Find dy/dx for x² + y² = 25:

F(x, y) = x² + y² - 25 = 0

F_x = 2x
F_y = 2y

dy/dx = -F_x / F_y = -2x / 2y = -x/y

At (3, 4): dy/dx = -3/4
```

---

### Case 2: F(x, y, z) = 0

If F(x, y, z) = 0 defines z implicitly as function of x and y:

```
∂z/∂x = -F_x / F_z
∂z/∂y = -F_y / F_z
```

### Example 5

```
Find ∂z/∂x and ∂z/∂y for x² + y² + z² = 1:

F(x, y, z) = x² + y² + z² - 1 = 0

F_x = 2x, F_y = 2y, F_z = 2z

∂z/∂x = -F_x / F_z = -2x / 2z = -x/z
∂z/∂y = -F_y / F_z = -2y / 2z = -y/z
```

---

### Case 3: System of Equations

For F(x, y, u, v) = 0 and G(x, y, u, v) = 0,
where u and v are functions of x and y:

Use **Jacobian determinants** (advanced topic).

---

## 4. Total Differential

### Definition

For z = f(x, y), the **total differential** is:

```
dz = (∂f/∂x)dx + (∂f/∂y)dy

Or: dz = f_x dx + f_y dy
```

**Meaning:** Approximate change in z due to small changes in x and y.

### Approximation

For small changes Δx and Δy:

```
Δz ≈ dz = f_x dx + f_y dy

where dx = Δx, dy = Δy
```

### Example 6

```
Estimate change in f(x, y) = x²y when (x, y) changes from (2, 3) to (2.1, 2.9):

f_x = 2xy  →  f_x(2,3) = 12
f_y = x²   →  f_y(2,3) = 4

dx = 2.1 - 2 = 0.1
dy = 2.9 - 3 = -0.1

df = 12(0.1) + 4(-0.1) = 1.2 - 0.4 = 0.8

Actual change:
f(2.1, 2.9) - f(2, 3) = (2.1)²(2.9) - (2)²(3)
                       = 12.789 - 12 = 0.789

Approximation: 0.8 (very close!) ✓
```

---

## 5. Error Propagation

### Formula

If z = f(x, y) and x, y have measurement errors dx, dy:

**Maximum error:**
```
|dz| ≤ |f_x|·|dx| + |f_y|·|dy|
```

**Relative error:**
```
|dz/z| ≈ |(f_x·dx + f_y·dy) / f|
```

### Example 7

```
Volume of cylinder: V = πr²h

Radius measured as r = 5 ± 0.1 cm
Height measured as h = 10 ± 0.2 cm

Find maximum error in volume:

∂V/∂r = 2πrh = 2π(5)(10) = 100π
∂V/∂h = πr² = π(25) = 25π

dV = (100π)(0.1) + (25π)(0.2)
   = 10π + 5π = 15π ≈ 47.1 cm³

V = π(5)²(10) = 250π ≈ 785.4 cm³

Relative error: 47.1/785.4 ≈ 6%
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, exp, sin, cos, ln, sqrt, Function, idiff, lambdify

x, y, z, t, s, u, v = symbols('x y z t s u v')

# === Chain Rule Examples ===

# Example 1: z = f(x, y), x = x(t), y = y(t)
print("=" * 60)
print("CHAIN RULE: One Intermediate Variable")
print("=" * 60)

z_expr = x**2 + y**2
x_t = t**2
y_t = t**3

# Partial derivatives
dz_dx = diff(z_expr, x)
dz_dy = diff(z_expr, y)

# Derivatives of x and y wrt t
dx_dt = diff(x_t, t)
dy_dt = diff(y_t, t)

# Chain rule
dz_dt_chain = dz_dx.subs({x: x_t, y: y_t}) * dx_dt + \
              dz_dy.subs({x: x_t, y: y_t}) * dy_dt

print(f"z = {z_expr}, x = {x_t}, y = {y_t}")
print(f"dz/dt (chain rule) = {dz_dt_chain}")

# Verify by direct substitution
z_direct = (x_t)**2 + **(y_t)2
dz_dt_direct = diff(z_direct, t)
print(f"dz/dt (direct) = {dz_dt_direct}")
print(f"Match: {dz_dt_chain.simplify() == dz_dt_direct.simplify()}")

# Example 2: z = f(x, y), x = x(s, t), y = y(s, t)
print("\n" + "=" * 60)
print("CHAIN RULE: Two Intermediate Variables")
print("=" * 60)

z_expr = exp(x) * cos(y)
x_st = s**2 - t**2
y_st = 2*s*t

# Partials
dz_dx = diff(z_expr, x)
dz_dy = diff(z_expr, y)

# Partials of x and y
dx_ds = diff(x_st, s)
dx_dt = diff(x_st, t)
dy_ds = diff(y_st, s)
dy_dt = diff(y_st, t)

# Chain rule
dz_ds = dz_dx.subs({x: x_st, y: y_st}) * dx_ds + \
        dz_dy.subs({x: x_st, y: y_st}) * dy_ds

dz_dt = dz_dx.subs({x: x_st, y: y_st}) * dx_dt + \
        dz_dy.subs({x: x_st, y: y_st}) * dy_dt

print(f"z = {z_expr}")
print(f"x = {x_st}, y = {y_st}")
print(f"∂z/∂s = {dz_ds}")
print(f"∂z/∂t = {dz_dt}")

# === Tree Diagram Visualization ===

def draw_chain_tree(dependencies):
    """
    Visualize chain rule dependencies as a tree.
    dependencies: dict like {'z': ['x', 'y'], 'x': ['s', 't'], 'y': ['s', 't']}
    """
    print("\nChain Rule Tree:")
    print("-" * 40)
    
    for var, deps in dependencies.items():
        for dep in deps:
            print(f"  {var}")
            print(f"   |")
            print(f"  {dep}")
            print()

# Example
deps = {'z': ['x', 'y'], 'x': ['s', 't'], 'y': ['s', 't']}
draw_chain_tree(deps)

# === Implicit Differentiation ===

print("\n" + "=" * 60)
print("IMPLICIT DIFFERENTIATION")
print("=" * 60)

# Example: x² + y² = 25
F = x**2 + y**2 - 25

# Using sympy's implicit differentiation
dy_dx_implicit = -diff(F, x) / diff(F, y)
print(f"F(x,y) = {F} = 0")
print(f"dy/dx = {dy_dx_implicit}")

# Verify at point (3, 4)
dy_dx_at_point = dy_dx_implicit.subs({x: 3, y: 4})
print(f"At (3, 4): dy/dx = {dy_dx_at_point}")

# === Total Differential ===

print("\n" + "=" * 60)
print("TOTAL DIFFERENTIAL")
print("=" * 60)

f = x**2 * y
df_dx = diff(f, x)
df_dy = diff(f, y)

print(f"f(x, y) = {f}")
print(f"df = ({df_dx})dx + ({df_dy})dy")

# Estimate change from (2, 3) to (2.1, 2.9)
x0, y0 = 2, 3
dx, dy = 0.1, -0.1

df_approx = df_dx.subs({x: x0, y: y0}) * dx + \
            df_dy.subs({x: x0, y: y0}) * dy

actual_change = f.subs({x: x0+dx, y: y0+dy}) - f.subs({x: x0, y: y0})

print(f"\nFrom ({x0}, {y0}) to ({x0+dx}, {y0+dy}):")
print(f"Approximate change (df): {df_approx}")
print(f"Actual change: {actual_change}")
print(f"Error: {abs(df_approx - actual_change)}")

# === Error Propagation ===

print("\n" + "=" * 60)
print("ERROR PROPAGATION")
print("=" * 60)

# Volume of cylinder
V = lambda r, h: np.pi * r**2 * h
dV_dr = lambda r, h: 2 * np.pi * r * h
dV_dh = lambda r, h: np.pi * r**2

r, h = 5, 10
dr, dh = 0.1, 0.2

# Maximum error
max_error = abs(dV_dr(r, h) * dr) + abs(dV_dh(r, h) * dh)

volume = V(r, h)
relative_error = max_error / volume * 100

print(f"Cylinder: r = {r} ± {dr} cm, h = {h} ± {dh} cm")
print(f"Volume = {volume:.2f} cm³")
print(f"Maximum error = ±{max_error:.2f} cm³")
print(f"Relative error = ±{relative_error:.2f}%")

# === Numerical Chain Rule Verification ===

def chain_rule_numerical():
    """Verify chain rule numerically"""
    
    # z = x² + y², x = t², y = t³
    z = lambda x, y: x**2 + y**2
    x = lambda t: t**2
    y = lambda t: t**3
    
    # Partials
    dz_dx = lambda x, y: 2*x
    dz_dy = lambda x, y: 2*y
    
    # Derivatives of x and y
    dx_dt = lambda t: 2*t
    dy_dt = lambda t: 3*t**2
    
    t_val = 2
    h = 1e-5
    
    # Chain rule prediction
    x_val = x(t_val)
    y_val = y(t_val)
    
    dz_dt_chain = dz_dx(x_val, y_val) * dx_dt(t_val) + \
                  dz_dy(x_val, y_val) * dy_dt(t_val)
    
    # Numerical derivative
    z_composite = lambda t: z(x(t), y(t))
    dz_dt_numerical = (z_composite(t_val + h) - z_composite(t_val - h)) / (2*h)
    
    print(f"\nNumerical verification at t = {t_val}:")
    print(f"Chain rule: dz/dt = {dz_dt_chain}")
    print(f"Numerical:  dz/dt = {dz_dt_numerical:.6f}")
    print(f"Match: {abs(dz_dt_chain - dz_dt_numerical) < 1e-4}")

chain_rule_numerical()

# === Backpropagation Simulation ===

print("\n" + "=" * 60)
print("BACKPROPAGATION SIMULATION")
print("=" * 60)

def simple_neural_network():
    """Simulate backpropagation using chain rule"""
    
    # Simple network: output = W2 * sigmoid(W1 * x)
    x = 2.0
    W1 = 0.5
    W2 = 1.0
    
    # Forward pass
    z1 = W1 * x           # weighted input
    a1 = 1 / (1 + np.exp(-z1))  # sigmoid activation
    output = W2 * a1      # output
    
    # Loss (MSE with target = 1)
    target = 1.0
    loss = 0.5 * (output - target)**2
    
    print(f"Forward pass:")
    print(f"  z1 = W1 * x = {W1} * {x} = {z1}")
    print(f"  a1 = sigmoid(z1) = {a1:.4f}")
    print(f"  output = W2 * a1 = {W2} * {a1:.4f} = {output:.4f}")
    print(f"  loss = 0.5 * (output - target)² = {loss:.6f}")
    
    # Backward pass (chain rule!)
    dloss_doutput = output - target
    doutput_dW2 = a1
    doutput_da1 = W2
    da1_dz1 = a1 * (1 - a1)  # sigmoid derivative
    dz1_dW1 = x
    
    # Chain rule for gradients
    dloss_dW2 = dloss_doutput * doutput_dW2
    dloss_da1 = dloss_doutput * doutput_da1
    dloss_dz1 = dloss_da1 * da1_dz1
    dloss_dW1 = dloss_dz1 * dz1_dW1
    
    print(f"\nBackward pass (chain rule):")
    print(f"  ∂L/∂W2 = ∂L/∂output · ∂output/∂W2 = {dloss_doutput:.4f} * {a1:.4f} = {dloss_dW2:.6f}")
    print(f"  ∂L/∂W1 = ∂L/∂z1 · ∂z1/∂W1 = ({dloss_da1:.4f} * {da1_dz1:.4f}) * {x} = {dloss_dW1:.6f}")

simple_neural_network()
```

---

## 📊 Summary Table

| Case | Formula | Tree |
|------|---------|------|
| **z=f(x,y), x=x(t), y=y(t)** | dz/dt = f_x·dx/dt + f_y·dy/dt | z→x,y→t |
| **z=f(x,y), x=x(s,t), y=y(s,t)** | ∂z/∂s = f_x·x_s + f_y·y_s | z→x,y→s,t |
| **w=f(x,y,z), all depend on t** | dw/dt = f_x·x' + f_y·y' + f_z·z' | w→x,y,z→t |
| **Implicit F(x,y)=0** | dy/dx = -F_x/F_y | - |
| **Implicit F(x,y,z)=0** | ∂z/∂x = -F_x/F_z, ∂z/∂y = -F_y/F_z | - |
| **Total differential** | dz = f_x dx + f_y dy | - |

---

## 🎯 ML Applications

| Application | Chain Rule Concept |
|-------------|-------------------|
| **Backpropagation** | Chain rule through computation graph |
| **Automatic Differentiation** | Systematic chain rule application |
| **Neural Network Training** | ∂L/∂W = ∂L/∂output · ∂output/∂W |
| **Gradient Computation** | Chain rule for composite functions |
| **Computational Graphs** | Tree diagrams for dependencies |

**Backpropagation as Chain Rule:**
```
Neural network: L = Loss(f(g(h(x))))

dL/dh = dL/df · df/dg · dg/dh

This IS the chain rule!
```

---

## ❓ Quick Check Questions

1. State the chain rule for z = f(x, y) where x = x(t), y = y(t).
2. How do you draw a tree diagram for chain rule?
3. What is implicit differentiation formula for F(x, y) = 0?
4. Define total differential.
5. How is error propagation calculated?
6. Explain how backpropagation uses chain rule.

---

## 📝 Answers to Quick Check

1. **Chain rule (one variable):**
   - dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)

2. **Tree diagram:**
   - Top: dependent variable
   - Middle: intermediate variables
   - Bottom: independent variables
   - Multiply along paths, add results

3. **Implicit differentiation:**
   - dy/dx = -F_x / F_y

4. **Total differential:**
   - dz = f_x dx + f_y dy
   - Approximate change in z

5. **Error propagation:**
   - |dz| ≤ |f_x|·|dx| + |f_y|·|dy|

6. **Backpropagation:**
   - Compute ∂L/∂W by chaining derivatives
   - ∂L/∂W = ∂L/∂output · ∂output/∂W
   - This is chain rule through the network

---

**Status:** ✅ Complete
**Next:** Optimization in Multiple Variables
