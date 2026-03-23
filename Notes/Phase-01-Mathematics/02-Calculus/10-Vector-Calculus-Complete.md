# 1.2.10 Vector Calculus (Essentials for ML)

## 🎯 Quick Overview
- **Vector fields**: Assign vector to each point
- **Gradient, divergence, curl**: Three fundamental operators
- **Line and surface integrals**: Integration over curves and surfaces
- **Foundation for**: Physics-informed ML, fluid dynamics, electromagnetism

---

## 1. Vector Fields

### Definition

A **vector field** assigns a vector to each point in space:

**2D:** F(x, y) = P(x, y)i + Q(x, y)j = ⟨P, Q⟩

**3D:** F(x, y, z) = P(x, y, z)i + Q(x, y, z)j + R(x, y, z)k = ⟨P, Q, R⟩

### Examples

| Field | Formula | Visualization |
|-------|---------|---------------|
| **Constant** | F = ⟨1, 0⟩ | All arrows same |
| **Radial** | F = ⟨x, y⟩ | Arrows point outward |
| **Rotational** | F = ⟨-y, x⟩ | Arrows circulate |
| **Gradient** | F = ∇f | Arrows point uphill |

### Example 1

```
F(x, y) = ⟨-y, x⟩ (rotation field)

At (1, 0): F = ⟨0, 1⟩ (points up)
At (0, 1): F = ⟨-1, 0⟩ (points left)
At (-1, 0): F = ⟨0, -1⟩ (points down)
At (0, -1): F = ⟨1, 0⟩ (points right)

This creates counterclockwise circulation!
```

---

## 2. Gradient (Review)

### Definition

For scalar field f(x, y, z):

```
∇f = ⟨∂f/∂x, ∂f/∂y, ∂f/∂z⟩
```

### Properties

| Property | Meaning |
|----------|---------|
| **Direction** | Points toward steepest increase |
| **Magnitude** | Rate of increase in that direction |
| **Perpendicular** | Orthogonal to level surfaces |

### Example 2

```
f(x, y, z) = x²y + yz²

∇f = ⟨2xy, x² + z², 2yz⟩

At (1, 2, 1):
∇f(1,2,1) = ⟨4, 2, 4⟩
```

---

## 3. Divergence

### Definition

For vector field F = ⟨P, Q, R⟩:

```
div F = ∇ · F = ∂P/∂x + ∂Q/∂y + ∂R/∂z
```

**Note:** This is a DOT product of ∇ with F.

### Physical Meaning

**Divergence measures:**
- Rate of "outflow" from a point
- Source strength (positive divergence)
- Sink strength (negative divergence)

| div F | Interpretation |
|-------|---------------|
| > 0 | Source (fluid expanding) |
| < 0 | Sink (fluid compressing) |
| = 0 | Incompressible (solenoidal) |

### Example 3

```
F(x, y, z) = ⟨x², y², z²⟩

div F = ∂(x²)/∂x + ∂(y²)/∂y + ∂(z²)/∂z
      = 2x + 2y + 2z

At (1, 1, 1): div F = 6 > 0 (source)
At (-1, -1, -1): div F = -6 < 0 (sink)
```

```
F(x, y, z) = ⟨-y, x, 0⟩ (rotation field)

div F = ∂(-y)/∂x + ∂(x)/∂y + ∂(0)/∂z
      = 0 + 0 + 0 = 0

This field is divergence-free (incompressible)
```

---

## 4. Curl

### Definition

For vector field F = ⟨P, Q, R⟩:

```
curl F = ∇ × F = | i     j     k    |
                | ∂/∂x  ∂/∂y  ∂/∂z |
                | P     Q     R    |

= ⟨∂R/∂y - ∂Q/∂z, ∂P/∂z - ∂R/∂x, ∂Q/∂x - ∂P/∂y⟩
```

**Note:** This is a CROSS product of ∇ with F.

### Physical Meaning

**Curl measures:**
- Rotation/tendency to swirl
- Axis of rotation (direction of curl)
- Strength of rotation (magnitude)

| curl F | Interpretation |
|--------|---------------|
| ≠ 0 | Rotational field |
| = 0 | Irrotational (conservative) |

### Example 4

```
F(x, y, z) = ⟨-y, x, 0⟩ (rotation in xy-plane)

curl F = | i     j     k    |
         | ∂/∂x  ∂/∂y  ∂/∂z |
         | -y    x     0    |

= ⟨∂(0)/∂y - ∂(x)/∂z, ∂(-y)/∂z - ∂(0)/∂x, ∂(x)/∂x - ∂(-y)/∂y⟩
= ⟨0 - 0, 0 - 0, 1 - (-1)⟩
= ⟨0, 0, 2⟩

Curl points in +z direction (right-hand rule)
Magnitude = 2 (strength of rotation)
```

```
F(x, y, z) = ⟨x, y, z⟩ (radial field)

curl F = ⟨∂z/∂y - ∂y/∂z, ∂x/∂z - ∂z/∂x, ∂y/∂x - ∂x/∂y⟩
       = ⟨0 - 0, 0 - 0, 0 - 0⟩
       = ⟨0, 0, 0⟩

Radial fields are irrotational (no curl)
```

---

## 5. Line Integrals

### Line Integral of Scalar Field

For scalar f along curve C:

```
∫_C f(x, y) ds = ∫ₐᵇ f(r(t)) · |r'(t)| dt

where r(t) parametrizes C for t ∈ [a, b]
```

**Application:** Mass of wire with density f.

### Line Integral of Vector Field

For vector field F along curve C:

```
∫_C F · dr = ∫ₐᵇ F(r(t)) · r'(t) dt

This is the WORK done by force F along path C
```

### Example 5

```
Compute ∫_C F · dr where F = ⟨-y, x⟩ and C is the unit circle

Parametrize: r(t) = ⟨cos(t), sin(t)⟩ for t ∈ [0, 2π]

r'(t) = ⟨-sin(t), cos(t)⟩
F(r(t)) = ⟨-sin(t), cos(t)⟩

F · r' = (-sin(t))(-sin(t)) + (cos(t))(cos(t))
       = sin²(t) + cos²(t) = 1

∫_C F · dr = ∫₀^{2π} 1 dt = 2π

Non-zero! This field is NOT conservative.
```

### Fundamental Theorem for Line Integrals

**If F = ∇f (conservative field):**

```
∫_C ∇f · dr = f(end) - f(start)

Line integral depends only on endpoints, not path!
```

**Consequence:** ∫_C ∇f · dr = 0 for any closed loop.

---

## 6. Surface Integrals

### Parametric Surfaces

Surface S parametrized by r(u, v):

```
r(u, v) = ⟨x(u,v), y(u,v), z(u,v)⟩
```

**Surface area element:**
```
dS = |r_u × r_v| du dv
```

### Surface Integral of Scalar Field

```
∬_S f(x, y, z) dS = ∬_D f(r(u,v)) · |r_u × r_v| du dv
```

**Application:** Mass of curved surface with density f.

### Surface Integral of Vector Field (Flux)

```
∬_S F · dS = ∬_S F · n dS

where n is the unit normal vector
```

**Physical meaning:** Rate of fluid flow through surface.

### Example 6

```
Find flux of F = ⟨x, y, z⟩ through unit sphere

Sphere: r(φ, θ) = ⟨sin(φ)cos(θ), sin(φ)sin(θ), cos(φ)⟩
        for φ ∈ [0, π], θ ∈ [0, 2π]

On sphere: F = r (radial field)
Normal vector: n = r/|r| = r (unit sphere)

F · n = r · r = 1

Flux = ∬_S 1 dS = Surface area of sphere = 4π
```

---

## 7. Green's Theorem (Conceptual)

### Statement

For vector field F = ⟨P, Q⟩ and region D with boundary C:

```
∮_C (P dx + Q dy) = ∬_D (∂Q/∂x - ∂P/∂y) dA

Line integral around boundary = Double integral of curl over region
```

### Example 7

```
Verify Green's theorem for F = ⟨-y, x⟩ over unit disk

Line integral (from Example 5): ∮_C F · dr = 2π

Double integral:
∂Q/∂x - ∂P/∂y = ∂(x)/∂x - ∂(-y)/∂y = 1 - (-1) = 2

∬_D 2 dA = 2 · (area of disk) = 2 · π = 2π  ✓
```

---

## 8. Stokes' Theorem (Conceptual)

### Statement

For vector field F and surface S with boundary curve C:

```
∮_C F · dr = ∬_S (curl F) · dS

Line integral around boundary = Surface integral of curl
```

**Generalization of Green's theorem to 3D.**

---

## 9. Divergence Theorem (Conceptual)

### Statement

For vector field F and solid region V with boundary surface S:

```
∯_S F · dS = ∭_V (div F) dV

Flux through boundary = Volume integral of divergence
```

### Example 8

```
Verify divergence theorem for F = ⟨x, y, z⟩ over unit ball

Volume integral:
div F = 3
∭_V 3 dV = 3 · (volume of sphere) = 3 · (4π/3) = 4π

Surface integral (from Example 6):
∯_S F · dS = 4π  ✓
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, Matrix, sqrt, sin, cos, pi, integrate
from mpl_toolkits.mplot3d import Axes3D

x, y, z, t, u, v = symbols('x y z t u v')

# === Vector Field Visualization ===

print("=" * 60)
print("VECTOR FIELDS")
print("=" * 60)

def plot_vector_field_2d(P, Q, x_range, y_range, density=15):
    """Plot 2D vector field"""
    x_vals = np.linspace(x_range[0], x_range[1], density)
    y_vals = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    U = P(X, Y)
    V = Q(X, Y)
    
    # Normalize for visualization
    N = np.sqrt(U**2 + V**2)
    U_norm = U / np.where(N > 0, N, 1)
    V_norm = V / np.where(N > 0, N, 1)
    
    plt.figure(figsize=(10, 8))
    
    # Quiver plot
    plt.quiver(X, Y, U_norm, V_norm, N, cmap='viridis', 
               scale=30, width=0.003, headwidth=4)
    
    plt.title('Vector Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.colorbar(label='Magnitude')
    plt.show()

# Example fields
print("\n1. Rotation field F = <-y, x>")
plot_vector_field_2d(
    lambda x, y: -y,
    lambda x, y: x,
    (-3, 3), (-3, 3)
)

print("\n2. Radial field F = <x, y>")
plot_vector_field_2d(
    lambda x, y: x,
    lambda x, y: y,
    (-3, 3), (-3, 3)
)

print("\n3. Source field F = <x², y²>")
plot_vector_field_2d(
    lambda x, y: x**2,
    lambda x, y: y**2,
    (-3, 3), (-3, 3)
)

# === Gradient, Divergence, Curl ===

print("\n" + "=" * 60)
print("GRADIENT, DIVERGENCE, CURL")
print("=" * 60)

# Define scalar field
f = x**2 * y + y * z**2

# Gradient
grad_f = Matrix([diff(f, var) for var in [x, y, z]])
print(f"\nScalar field: f = {f}")
print(f"Gradient: ∇f = {grad_f}")

# Define vector field
F = Matrix([x**2, y**2, z**2])

# Divergence
div_F = diff(F[0], x) + diff(F[1], y) + diff(F[2], z)
print(f"\nVector field: F = {F}")
print(f"Divergence: ∇·F = {div_F}")

# Curl
curl_F = Matrix([
    diff(F[2], y) - diff(F[1], z),
    diff(F[0], z) - diff(F[2], x),
    diff(F[1], x) - diff(F[0], y)
])
print(f"Curl: ∇×F = {curl_F}")

# Rotation field
F_rot = Matrix([-y, x, 0])
curl_F_rot = Matrix([
    diff(F_rot[2], y) - diff(F_rot[1], z),
    diff(F_rot[0], z) - diff(F_rot[2], x),
    diff(F_rot[1], x) - diff(F_rot[0], y)
])
print(f"\nRotation field: F = {-y, x, 0}")
print(f"Curl: ∇×F = {curl_F_rot}")
print(f"This field has constant curl in z-direction!")

# === Line Integral ===

print("\n" + "=" * 60)
print("LINE INTEGRALS")
print("=" * 60)

# Line integral of F = <-y, x> along unit circle
F_x = -y
F_y = x

# Parametrization: r(t) = <cos(t), sin(t)>
x_t = cos(t)
y_t = sin(t)

# Derivatives
dx_dt = diff(x_t, t)
dy_dt = diff(y_t, t)

# F along curve
F_x_t = F_x.subs({x: x_t, y: y_t})
F_y_t = F_y.subs({x: x_t, y: y_t})

# Dot product F · r'
integrand = F_x_t * dx_dt + F_y_t * dy_dt

print(f"F = <-y, x>")
print(f"Path: unit circle r(t) = <cos(t), sin(t)>")
print(f"F · r' = {integrand}")
print(f"Simplified: {integrand.simplify()}")

# Integrate
line_integral = integrate(integrand.simplify(), (t, 0, 2*pi))
print(f"\n∮_C F · dr = {line_integral}")
print(f"This is non-zero → field is NOT conservative")

# === Conservative Field Test ===

print("\n" + "=" * 60)
print("CONSERVATIVE FIELD TEST")
print("=" * 60)

def is_conservative_2d(P_expr, Q_expr):
    """Check if 2D field is conservative"""
    dQ_dx = diff(Q_expr, x)
    dP_dy = diff(P_expr, y)
    
    print(f"F = <{P_expr}, {Q_expr}>")
    print(f"∂Q/∂x = {dQ_dx}")
    print(f"∂P/∂y = {dP_dy}")
    
    if dQ_dx == dP_dy:
        print("✓ Field is CONSERVATIVE (∂Q/∂x = ∂P/∂y)")
        return True
    else:
        print("✗ Field is NOT conservative")
        return False

# Test fields
print("\nTest 1: F = <y, x>")
is_conservative_2d(y, x)

print("\nTest 2: F = <-y, x>")
is_conservative_2d(-y, x)

print("\nTest 3: F = <2xy, x²>")
is_conservative_2d(2*x*y, x**2)

# === Surface Integral (Flux) ===

print("\n" + "=" * 60)
print("SURFACE INTEGRALS")
print("=" * 60)

# Flux through sphere
# F = <x, y, z>, sphere of radius R

R = symbols('R', positive=True)

# Parametrization of sphere
x_sphere = R * sin(phi) * cos(theta)
y_sphere = R * sin(phi) * sin(theta)
z_sphere = R * cos(phi)

# Normal vector magnitude (for sphere): R² sin(φ)
# F · n = R (on surface of sphere)

# Flux = ∬ R · R² sin(φ) dφ dθ
flux_integrand = R**3 * sin(phi)
flux = integrate(integrate(flux_integrand, (phi, 0, pi)), (theta, 0, 2*pi))

print(f"Flux of F = <x, y, z> through sphere of radius R:")
print(f"Flux = {flux}")
print(f"For R=1: Flux = {flux.subs(R, 1)} = 4π")

# === Numerical Line Integral ===

def line_integral_numerical(F, r, t_range, n=1000):
    """Compute line integral numerically"""
    t_min, t_max = t_range
    dt = (t_max - t_min) / n
    
    total = 0
    for i in range(n):
        t_mid = t_min + (i + 0.5) * dt
        
        # Position and derivative
        pos = r(t_mid)
        dr_dt = (r(t_mid + 1e-6) - r(t_mid - 1e-6)) / (2e-6)
        
        # F at position
        F_val = F(pos)
        
        # Dot product
        total += np.dot(F_val, dr_dt) * dt
    
    return total

# Example: F = <-y, x> along unit circle
F = lambda r: np.array([-r[1], r[0]])
r = lambda t: np.array([np.cos(t), np.sin(t)])

result = line_integral_numerical(F, r, (0, 2*np.pi))
print(f"\nNumerical line integral of F = <-y, x> along unit circle:")
print(f"Result: {result:.6f}")
print(f"Exact: {2*np.pi:.6f}")
print(f"Error: {abs(result - 2*np.pi):.2e}")

# === Gradient Field Visualization ===

def plot_gradient_field(f, x_range, y_range, density=20):
    """Plot scalar field with gradient vectors"""
    x_vals = np.linspace(x_range[0], x_range[1], density)
    y_vals = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute function values
    Z = f(X, Y)
    
    # Compute gradient numerically
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    dZ_dx, dZ_dy = np.gradient(Z, dx, dy)
    
    plt.figure(figsize=(12, 5))
    
    # Contour plot
    plt.subplot(1, 2, 1)
    contour = plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    plt.contour(X, Y, Z, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    plt.title(f'f(x,y) = {f.__name__}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(contour, label='f(x,y)')
    plt.grid(True, alpha=0.3)
    
    # Gradient field
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.3)
    plt.quiver(X, Y, dZ_dx, dZ_dy, color='red', scale=30, width=0.003)
    plt.title('Gradient Field ∇f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Example
f = lambda x, y: x**2 - y**2
plot_gradient_field(f, (-3, 3), (-3, 3))
```

---

## 📊 Summary Table

| Operator | Symbol | Formula | Type |
|----------|--------|---------|------|
| **Gradient** | ∇f | ⟨∂f/∂x, ∂f/∂y, ∂f/∂z⟩ | Scalar → Vector |
| **Divergence** | ∇·F | ∂P/∂x + ∂Q/∂y + ∂R/∂z | Vector → Scalar |
| **Curl** | ∇×F | ⟨∂R/∂y-∂Q/∂z, ...⟩ | Vector → Vector |
| **Line Integral** | ∫_C F·dr | Work along curve | - |
| **Surface Integral** | ∬_S F·dS | Flux through surface | - |

**Theorems:**
| Theorem | Formula |
|---------|---------|
| **Green's** | ∮_C F·dr = ∬_D (curl F)·k dA |
| **Stokes'** | ∮_C F·dr = ∬_S (curl F)·dS |
| **Divergence** | ∯_S F·dS = ∭_V div F dV |

---

## 🎯 ML Applications

| Application | Vector Calculus Concept |
|-------------|------------------------|
| **Gradient Descent** | Follow -∇L (negative gradient) |
| **Physics-Informed NN** | PDEs with divergence, curl |
| **Fluid Dynamics ML** | Navier-Stokes (div, curl) |
| **Electromagnetism** | Maxwell's equations |
| **Continuity Equations** | Divergence theorem |
| **Conservative Systems** | Curl-free fields |

---

## ❓ Quick Check Questions

1. What does the gradient vector represent?
2. What is the physical meaning of divergence?
3. What does curl measure?
4. What is a conservative field?
5. State the fundamental theorem for line integrals.
6. How are Green's, Stokes', and Divergence theorems related?

---

## 📝 Answers to Quick Check

1. **Gradient:**
   - Direction of steepest increase
   - Magnitude = maximum rate of change

2. **Divergence:**
   - Source/sink strength
   - Rate of expansion/compression

3. **Curl:**
   - Rotation tendency
   - Axis and strength of swirl

4. **Conservative field:**
   - F = ∇f for some potential f
   - curl F = 0
   - Path-independent line integrals

5. **Fundamental theorem for line integrals:**
   - ∫_C ∇f·dr = f(end) - f(start)

6. **Theorems relationship:**
   - All relate boundary integral to interior integral
   - Green's: 2D special case of Stokes'
   - All generalize FTC to higher dimensions

---

**Status:** ✅ Complete
**Next:** Practice Problems (already created)

---

**CALCULUS COMPLETE!** 🎉

All 11 topic files + README + Practice Problems created.
