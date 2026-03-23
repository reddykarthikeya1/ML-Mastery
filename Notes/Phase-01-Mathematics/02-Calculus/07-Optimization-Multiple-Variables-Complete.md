# 1.2.7 Optimization in Multiple Variables

## 🎯 Quick Overview
- **Critical points**: Where ∇f = 0
- **Hessian matrix**: Second derivatives for classification
- **Lagrange multipliers**: Constrained optimization
- **Foundation for**: ML optimization, constrained learning

---

## 1. Critical Points in Multiple Variables

### Definition

A point **(a, b)** is a **critical point** of f(x, y) if:

1. **∇f(a, b) = 0** (both partials are zero), OR
2. **∇f(a, b) does not exist** (partial undefined)

### Finding Critical Points

**Step 1:** Compute f_x and f_y
**Step 2:** Solve the system:
```
f_x(x, y) = 0
f_y(x, y) = 0
```
**Step 3:** Check where partials don't exist

### Example 1

```
f(x, y) = x³ + y³ - 3xy

f_x = 3x² - 3y
f_y = 3y² - 3x

Set both to zero:
3x² - 3y = 0  →  y = x²
3y² - 3x = 0  →  x = y²

Substitute: x = (x²)² = x⁴
x⁴ - x = 0
x(x³ - 1) = 0

x = 0 or x = 1

Critical points: (0, 0) and (1, 1)
```

---

## 2. Second Partial Derivative Test

### Theorem

Let (a, b) be a critical point where ∇f(a, b) = 0.

Define the **discriminant**:
```
D = f_xx(a,b) · f_yy(a,b) - [f_xy(a,b)]²

Or using determinant: D = det(H) where H is the Hessian
```

**Classification:**

| D | f_xx | Conclusion |
|---|------|------------|
| D > 0 | f_xx > 0 | **Local Minimum** |
| D > 0 | f_xx < 0 | **Local Maximum** |
| D < 0 | any | **Saddle Point** |
| D = 0 | any | **Test Inconclusive** |

### Example 2

```
Classify critical points of f(x, y) = x³ + y³ - 3xy:

Critical points: (0, 0) and (1, 1)

Second derivatives:
f_xx = 6x
f_yy = 6y
f_xy = -3

D = f_xx · f_yy - (f_xy)² = (6x)(6y) - 9 = 36xy - 9

At (0, 0):
D = 36(0)(0) - 9 = -9 < 0 → **Saddle Point**

At (1, 1):
D = 36(1)(1) - 9 = 27 > 0
f_xx(1,1) = 6 > 0 → **Local Minimum**

f(1,1) = 1 + 1 - 3 = -1

Local minimum value is -1 at (1, 1)
```

### Example 3

```
f(x, y) = x⁴ + y⁴ - 4xy

f_x = 4x³ - 4y = 0  →  y = x³
f_y = 4y³ - 4x = 0  →  x = y³

Substitute: x = (x³)³ = x⁹
x⁹ - x = 0
x(x⁸ - 1) = 0

x = 0, x = 1, x = -1

Critical points: (0, 0), (1, 1), (-1, -1)

Second derivatives:
f_xx = 12x²
f_yy = 12y²
f_xy = -4

D = 144x²y² - 16

At (0, 0): D = -16 < 0 → Saddle Point
At (1, 1): D = 144 - 16 = 128 > 0, f_xx = 12 > 0 → Local Minimum
At (-1, -1): D = 144 - 16 = 128 > 0, f_xx = 12 > 0 → Local Minimum
```

---

## 3. Hessian Matrix

### Definition

The **Hessian matrix** of f(x, y) is:

```
        [ f_xx   f_xy ]
H(f) =  |            |
        [ f_yx   f_yy ]
```

**For n variables:**
```
        [ ∂²f/∂x₁²    ∂²f/∂x₁∂x₂   ...   ∂²f/∂x₁∂xₙ ]
        | ∂²f/∂x₂∂x₁   ∂²f/∂x₂²    ...   ∂²f/∂x₂∂xₙ |
H(f) =  |    ...         ...      ...      ...     |
        [ ∂²f/∂xₙ∂x₁   ∂²f/∂xₙ∂x₂   ...   ∂²f/∂xₙ²  ]
```

### Classification Using Hessian

| Hessian Property | Conclusion |
|-----------------|------------|
| Positive definite (all eigenvalues > 0) | Local Minimum |
| Negative definite (all eigenvalues < 0) | Local Maximum |
| Indefinite (mixed eigenvalues) | Saddle Point |
| Singular (zero eigenvalue) | Test inconclusive |

### For 2×2 Hessian

```
H = [ a   b ]
    [ b   c ]

Positive definite:  a > 0 AND det(H) = ac - b² > 0
Negative definite:  a < 0 AND det(H) = ac - b² > 0
Indefinite:         det(H) < 0
```

### Example 4

```
f(x, y) = x² + 2xy + 3y²

H = [ 2   2 ]
    [ 2   6 ]

det(H) = 12 - 4 = 8 > 0
f_xx = 2 > 0

→ Positive definite → Local Minimum

Actually, this is a CONVEX function (global minimum)
```

---

## 4. Constrained Optimization

### Problem Formulation

**Maximize/Minimize:** f(x, y)
**Subject to:** g(x, y) = c

### Method of Lagrange Multipliers

**Step 1:** Define Lagrangian:
```
L(x, y, λ) = f(x, y) - λ(g(x, y) - c)
```

**Step 2:** Solve the system:
```
∂L/∂x = 0
∂L/∂y = 0
∂L/∂λ = 0  (this gives the constraint)

Equivalently:
∇f = λ∇g
g(x, y) = c
```

**Step 3:** Evaluate f at all solutions to find max/min

### Geometric Interpretation

```
At optimum: ∇f is parallel to ∇g

∇f = λ∇g

The level curve of f is tangent to constraint curve
```

### Example 5: Basic Lagrange

```
Maximize f(x, y) = xy
Subject to x² + y² = 1

Lagrangian: L(x, y, λ) = xy - λ(x² + y² - 1)

Partial derivatives:
∂L/∂x = y - 2λx = 0  →  y = 2λx
∂L/∂y = x - 2λy = 0  →  x = 2λy
∂L/∂λ = -(x² + y² - 1) = 0  →  x² + y² = 1

From first two equations:
y = 2λx and x = 2λy

Substitute: y = 2λ(2λy) = 4λ²y

If y ≠ 0: 4λ² = 1 → λ = ±1/2

Case λ = 1/2: y = x
  Constraint: x² + x² = 1 → x = ±1/√2
  Points: (1/√2, 1/√2), (-1/√2, -1/√2)

Case λ = -1/2: y = -x
  Constraint: x² + x² = 1 → x = ±1/√2
  Points: (1/√2, -1/√2), (-1/√2, 1/√2)

Evaluate f = xy:
f(1/√2, 1/√2) = 1/2  (MAXIMUM)
f(-1/√2, -1/√2) = 1/2  (MAXIMUM)
f(1/√2, -1/√2) = -1/2  (MINIMUM)
f(-1/√2, 1/√2) = -1/2  (MINIMUM)
```

---

## 5. Multiple Constraints

### Problem

**Maximize/Minimize:** f(x, y, z)
**Subject to:** g(x, y, z) = c₁ AND h(x, y, z) = c₂

### Lagrangian with Two Constraints

```
L(x, y, z, λ, μ) = f(x, y, z) - λ(g(x, y, z) - c₁) - μ(h(x, y, z) - c₂)
```

### System to Solve

```
∇f = λ∇g + μ∇h
g(x, y, z) = c₁
h(x, y, z) = c₂
```

### Example 6

```
Maximize f(x, y, z) = xyz
Subject to: x² + y² + z² = 1 AND x + y + z = 0

Lagrangian: L = xyz - λ(x² + y² + z² - 1) - μ(x + y + z)

System:
∂L/∂x = yz - 2λx - μ = 0
∂L/∂y = xz - 2λy - μ = 0
∂L/∂z = xy - 2λz - μ = 0
x² + y² + z² = 1
x + y + z = 0

(This requires solving a system of 5 equations in 5 unknowns)
```

---

## 6. KKT Conditions (Introduction)

### Karush-Kuhn-Tucker Conditions

For **inequality constraints**:

**Minimize:** f(x)
**Subject to:** gᵢ(x) ≤ 0, i = 1, ..., m

### KKT Conditions

At optimal point x*, there exist λ₁, ..., λₘ such that:

1. **Stationarity:** ∇f(x*) + Σλᵢ∇gᵢ(x*) = 0
2. **Primal feasibility:** gᵢ(x*) ≤ 0
3. **Dual feasibility:** λᵢ ≥ 0
4. **Complementary slackness:** λᵢgᵢ(x*) = 0

### Interpretation

- **Complementary slackness:** Either constraint is active (gᵢ = 0) or λᵢ = 0
- Only **active constraints** have non-zero multipliers

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solve, lambdify, Matrix, exp, ln, sqrt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

x, y, z, lam, mu = symbols('x y z lambda mu')

# === Finding Critical Points ===

print("=" * 60)
print("FINDING CRITICAL POINTS")
print("=" * 60)

f = x**3 + y**3 - 3*x*y

# First derivatives
f_x = diff(f, x)
f_y = diff(f, y)

print(f"f(x, y) = {f}")
print(f"f_x = {f_x}")
print(f"f_y = {f_y}")

# Solve for critical points
critical_points = solve([f_x, f_y], [x, y])
print(f"\nCritical points: {critical_points}")

# === Second Derivative Test ===

print("\n" + "=" * 60)
print("SECOND DERIVATIVE TEST")
print("=" * 60)

# Second derivatives
f_xx = diff(f_x, x)
f_yy = diff(f_y, y)
f_xy = diff(f_x, y)

print(f"f_xx = {f_xx}")
print(f"f_yy = {f_yy}")
print(f"f_xy = {f_xy}")

# Discriminant
D = f_xx * f_yy - f_xy**2
print(f"\nDiscriminant D = f_xx * f_yy - f_xy² = {D}")

# Classify each critical point
for point in critical_points:
    if isinstance(point, tuple):
        px, py = point
        D_val = D.subs({x: px, y: py})
        f_xx_val = f_xx.subs({x: px, y: py})
        f_val = f.subs({x: px, y: py})
        
        if D_val < 0:
            classification = "SADDLE POINT"
        elif D_val > 0:
            if f_xx_val > 0:
                classification = "LOCAL MINIMUM"
            else:
                classification = "LOCAL MAXIMUM"
        else:
            classification = "TEST INCONCLUSIVE"
        
        print(f"\nPoint ({px}, {py}):")
        print(f"  D = {D_val}")
        print(f"  f_xx = {f_xx_val}")
        print(f"  f = {f_val}")
        print(f"  Classification: {classification}")

# === Hessian Matrix ===

print("\n" + "=" * 60)
print("HESSIAN MATRIX")
print("=" * 60)

# Construct Hessian
H = Matrix([[f_xx, f_xy], [f_xy, f_yy]])
print(f"Hessian H = {H}")

# Eigenvalues at critical points
for point in critical_points:
    if isinstance(point, tuple):
        px, py = point
        H_at_point = H.subs({x: px, y: py})
        eigenvals = H_at_point.eigenvals()
        
        print(f"\nAt ({px}, {py}):")
        print(f"  H = {H_at_point}")
        print(f"  Eigenvalues: {list(eigenvals.keys())}")

# === Lagrange Multipliers ===

print("\n" + "=" * 60)
print("LAGRANGE MULTIPLIERS")
print("=" * 60)

# Maximize f = xy subject to x² + y² = 1
f_obj = x * y
g_constraint = x**2 + y**2 - 1

# Lagrangian
L = f_obj - lam * g_constraint

# Derivatives
L_x = diff(L, x)
L_y = diff(L, y)
L_lam = diff(L, lam)

print(f"Maximize: f = {f_obj}")
print(f"Subject to: {g_constraint} = 0")
print(f"\nLagrangian: L = {L}")
print(f"\nSystem of equations:")
print(f"  ∂L/∂x = {L_x} = 0")
print(f"  ∂L/∂y = {L_y} = 0")
print(f"  ∂L/∂λ = {L_lam} = 0")

# Solve
solutions = solve([L_x, L_y, L_lam], [x, y, lam])
print(f"\nSolutions: {solutions}")

# Evaluate objective at each solution
print("\nObjective values:")
for sol in solutions:
    if isinstance(sol, tuple):
        x_val, y_val, lam_val = sol
        obj_val = f_obj.subs({x: x_val, y: y_val})
        print(f"  ({x_val}, {y_val}): f = {obj_val}")

# === Numerical Optimization ===

print("\n" + "=" * 60)
print("NUMERICAL OPTIMIZATION")
print("=" * 60)

# Define function and constraint
def f_numeric(point):
    x, y = point
    return -(x**3 + y**3 - 3*x*y)  # Negative for minimization

def constraint(point):
    x, y = point
    return x**2 + y**2 - 1

# Initial guess
x0 = [0.5, 0.5]

# Constraints
cons = [{'type': 'eq', 'fun': constraint}]

# Optimize
result = minimize(f_numeric, x0, constraints=cons, method='SLSQP')

print(f"Optimization result:")
print(f"  x = {result.x}")
print(f"  f(x) = {-result.fun}")
print(f"  Success: {result.success}")
print(f"  Message: {result.message}")

# === Visualization of Critical Points ===

def plot_critical_points(f_expr, x_range, y_range):
    """Plot function with critical points marked"""
    
    f_func = lambdify([x, y], f_expr, 'numpy')
    
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_func(X, Y)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title(f'f(x,y) = {f_expr}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Contour with critical points
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    
    # Mark critical points
    for point in critical_points:
        if isinstance(point, tuple):
            px, py = point
            pz = f_func(px, py)
            ax2.plot(px, py, pz, 'r*', markersize=20)
            ax2.annotate(f'({px}, {py})', (px, py), 
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax2.set_title('Contour Plot with Critical Points')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(contour, ax=ax2, label='z')
    
    plt.tight_layout()
    plt.show()

plot_critical_points(f, (-2, 2), (-2, 2))

# === Constrained Optimization Visualization ===

def plot_constrained_optimization():
    """Visualize constrained optimization"""
    
    # Objective: f = xy
    # Constraint: x² + y² = 1
    
    x_vals = np.linspace(-1.5, 1.5, 400)
    y_vals = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = X * Y
    
    # Constraint circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    circle_z = circle_x * circle_y
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    ax1.plot(circle_x, circle_y, circle_z, 'r-', linewidth=3, label='Constraint')
    ax1.set_title('Objective with Constraint Curve')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    ax2.plot(circle_x, circle_y, 'r-', linewidth=3, label='Constraint: x²+y²=1')
    
    # Mark optimal points
    ax2.plot(1/np.sqrt(2), 1/np.sqrt(2), 'y*', markersize=20, label='Maximum')
    ax2.plot(-1/np.sqrt(2), -1/np.sqrt(2), 'y*', markersize=20)
    ax2.plot(1/np.sqrt(2), -1/np.sqrt(2), 'm*', markersize=20, label='Minimum')
    ax2.plot(-1/np.sqrt(2), 1/np.sqrt(2), 'm*', markersize=20)
    
    ax2.set_title('Constrained Optimization')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.legend()
    plt.colorbar(contour, ax=ax2, label='f(x,y) = xy')
    
    plt.tight_layout()
    plt.show()

plot_constrained_optimization()

# === Gradient Descent with Constraints ===

def projected_gradient_descent(f, grad_f, constraint_proj, x0, n_iter=100, lr=0.1):
    """Gradient descent with projection onto constraint set"""
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    
    for _ in range(n_iter):
        # Gradient step
        x = x - lr * grad_f(x)
        
        # Project onto constraint
        x = constraint_proj(x)
        
        path.append(x.copy())
    
    return np.array(path)

# Example: Minimize x² + y² subject to x + y = 1
f = lambda x: x[0]**2 + x[1]**2
grad_f = lambda x: np.array([2*x[0], 2*x[1]])

# Projection onto x + y = 1
def project_constraint(x):
    # Project point onto line x + y = 1
    # Formula: proj = x - (n·x - c) * n / |n|² where n = (1, 1), c = 1
    n = np.array([1, 1])
    c = 1
    return x - (np.dot(n, x) - c) * n / np.dot(n, n)

path = projected_gradient_descent(f, grad_f, project_constraint, x0=[0, 0], n_iter=50, lr=0.1)

print(f"\nProjected gradient descent:")
print(f"Final point: {path[-1]}")
print(f"Final value: {f(path[-1]):.6f}")
print(f"Constraint satisfied: {abs(sum(path[-1]) - 1) < 1e-6}")

# Plot path
plt.figure(figsize=(8, 8))
plt.plot(path[:, 0], path[:, 1], 'b-o', linewidth=2, markersize=5, label='Path')
plt.plot([0, 1], [1, 0], 'r-', linewidth=3, label='Constraint: x + y = 1')
plt.plot(path[0, 0], path[0, 1], 'go', markersize=15, label='Start')
plt.plot(path[-1, 0], path[-1, 1], 'y*', markersize=20, label='End')
plt.title('Projected Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

---

## 📊 Summary Table

| Concept | Formula | Use Case |
|---------|---------|----------|
| **Critical Point** | ∇f = 0 | Find extrema candidates |
| **Discriminant** | D = f_xx·f_yy - f_xy² | Classify critical points |
| **Hessian** | H = [[f_xx, f_xy], [f_yx, f_yy]] | Second-order information |
| **Lagrangian** | L = f - λ(g - c) | Constrained optimization |
| **KKT Conditions** | ∇f + Σλᵢ∇gᵢ = 0, λᵢ ≥ 0, λᵢgᵢ = 0 | Inequality constraints |

---

## 🎯 ML Applications

| Application | Optimization Concept |
|-------------|---------------------|
| **Gradient Descent** | Finding ∇L = 0 |
| **Constrained Optimization** | Lagrange multipliers |
| **Regularization** | Constrained loss minimization |
| **SVM** | Quadratic programming with constraints |
| **Neural Network Training** | Non-convex optimization |
| **Adam Optimizer** | Adaptive learning rates |

---

## ❓ Quick Check Questions

1. How do you find critical points in multivariable calculus?
2. What is the second derivative test?
3. What does the Hessian tell you?
4. Explain Lagrange multipliers geometrically.
5. What are KKT conditions?
6. When is a critical point a saddle point?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Finding critical points:**
   - Solve ∇f = 0 (all partials = 0)
   - Check where partials don't exist

2. **Second derivative test:**
   - D = f_xx·f_yy - f_xy²
   - D > 0, f_xx > 0: minimum
   - D > 0, f_xx < 0: maximum
   - D < 0: saddle

3. **Hessian tells:**
   - Curvature information
   - Positive definite → minimum
   - Negative definite → maximum
   - Indefinite → saddle

4. **Lagrange multipliers (geometric):**
   - At optimum, ∇f parallel to ∇g
   - Level curves tangent

5. **KKT conditions:**
   - Generalization of Lagrange for inequalities
   - Stationarity, feasibility, complementary slackness

6. **Saddle point when:**
   - D < 0 (Hessian indefinite)
   - Some directions increase, some decrease

</details>
---

**Status:** ✅ Complete
**Next:** Integration Fundamentals
