# 1.2.9 Multiple Integration

## 🎯 Quick Overview
- **Double integral**: Volume under surface
- **Iterated integral**: Evaluate one variable at a time
- **Change of variables**: Polar, cylindrical, spherical coordinates
- **Foundation for**: Probability (joint distributions), expectation, physics

---

## 1. Double Integrals Over Rectangles

### Definition

For f(x, y) defined on rectangle R = [a, b] × [c, d]:

```
∬_R f(x, y) dA = lim ΣΣ f(xᵢⱼ*, yᵢⱼ*)·ΔA
                 m,n→∞

where ΔA = Δx·Δy is the area element
```

**Geometric meaning:** Volume under surface z = f(x, y) over rectangle R.

### Computing Double Integrals (Fubini's Theorem)

**If f is continuous on R:**
```
∬_R f(x, y) dA = ∫ₐᵇ ∫_c^d f(x, y) dy dx
               = ∫_c^d ∫ₐᵇ f(x, y) dx dy
```

**Order doesn't matter for continuous functions on rectangles!**

### Example 1

```
Compute ∬_R (x² + y²) dA where R = [0, 1] × [0, 2]

Method 1 (integrate y first):
    1   2                1                    1
∫ ∫ (x² + y²) dy dx = ∫ [x²y + y³/3]₀² dx = ∫ (2x² + 8/3) dx
    0   0                0                    0
                      = [2x³/3 + 8x/3]₀¹ = 2/3 + 8/3 = 10/3

Method 2 (integrate x first):
    2   1                2                    2
∫ ∫ (x² + y²) dx dy = ∫ [x³/3 + y²x]₀¹ dy = ∫ (1/3 + y²) dy
    0   0                0                    0
                      = [y/3 + y³/3]₀² = 2/3 + 8/3 = 10/3  ✓
```

---

## 2. Double Integrals Over General Regions

### Types of Regions

**Type I (vertically simple):**
```
D = {(x, y) : a ≤ x ≤ b, g₁(x) ≤ y ≤ g₂(x)}

∬_D f(x, y) dA = ∫ₐᵇ ∫_{g₁(x)}^{g₂(x)} f(x, y) dy dx
```

**Type II (horizontally simple):**
```
D = {(x, y) : c ≤ y ≤ d, h₁(y) ≤ x ≤ h₂(y)}

∬_D f(x, y) dA = ∫_c^d ∫_{h₁(y)}^{h₂(y)} f(x, y) dx dy
```

### Example 2

```
Evaluate ∬_D xy dA where D is bounded by y = x and y = x²

Step 1: Find intersection points
x = x² → x(1-x) = 0 → x = 0, 1
Points: (0, 0) and (1, 1)

Step 2: Describe region
Type I: D = {(x, y) : 0 ≤ x ≤ 1, x² ≤ y ≤ x}

Step 3: Set up integral
    1   x              1                      1
∫ ∫ xy dy dx = ∫ [xy²/2]_{x²}^{x} dx = ∫ (x³/2 - x⁵/2) dx
    0   x²             0                      0
              = [x⁴/8 - x⁶/12]₀¹ = 1/8 - 1/12 = 1/24
```

### Example 3 (Changing Order)

```
Evaluate ∫₀¹ ∫_x¹ e^(y²) dy dx

Problem: ∫ e^(y²) dy has no elementary antiderivative!

Solution: Change order of integration

Original region: 0 ≤ x ≤ 1, x ≤ y ≤ 1

Sketch: Triangle with vertices (0,0), (1,1), (0,1)

New description: 0 ≤ y ≤ 1, 0 ≤ x ≤ y

New integral:
    1   y              1                      1
∫ ∫ e^(y²) dx dy = ∫ [x·e^(y²)]₀^y dy = ∫ y·e^(y²) dy
    0   0              0                      0

Now use substitution: u = y², du = 2y dy
    1                        1
∫ y·e^(y²) dy = (1/2) ∫ eᵘ du = (1/2)[eᵘ]₁⁰ = (1/2)(e - 1)
    0                        0
```

---

## 3. Double Integrals in Polar Coordinates

### Polar Coordinate Transformation

```
x = r·cos(θ)
y = r·sin(θ)

dA = r dr dθ  (Jacobian = r)
```

### When to Use Polar

- Circular regions
- Regions with radial symmetry
- Functions involving x² + y²

### Example 4

```
Find volume under z = 4 - x² - y² over the disk x² + y² ≤ 4

In polar:
x² + y² = r²
Region: 0 ≤ r ≤ 2, 0 ≤ θ ≤ 2π

Volume = ∬_D (4 - r²) r dr dθ
       = ∫₀^{2π} ∫₀² (4r - r³) dr dθ
       = ∫₀^{2π} [2r² - r⁴/4]₀² dθ
       = ∫₀^{2π} (8 - 4) dθ
       = ∫₀^{2π} 4 dθ
       = 4 · 2π = 8π
```

### Example 5 (Gaussian Integral)

```
Prove: ∫_{-∞}^{∞} e^(-x²) dx = √π

Consider: I = ∫_{-∞}^{∞} ∫_{-∞}^{∞} e^(-(x²+y²)) dx dy

In polar:
    ∞   2π                    ∞
I = ∫ ∫ e^(-r²) r dr dθ = ∫ dθ · ∫ e^(-r²) r dr
    0   0                     0   0

Inner integral (substitute u = r²):
    ∞                    ∞
∫ e^(-r²) r dr = (1/2) ∫ e^(-u) du = 1/2
    0                    0

So: I = 2π · (1/2) = π

But also: I = (∫_{-∞}^{∞} e^(-x²) dx)²

Therefore: ∫_{-∞}^{∞} e^(-x²) dx = √π
```

---

## 4. Triple Integrals

### Definition

For f(x, y, z) defined on box B = [a, b] × [c, d] × [p, q]:

```
∭_B f(x, y, z) dV = ∫ₐᵇ ∫_c^d ∫_p^q f(x, y, z) dz dy dx
```

### Applications

| Application | Formula |
|-------------|---------|
| Volume | V = ∭_D 1 dV |
| Mass (density ρ) | M = ∭_D ρ(x,y,z) dV |
| Center of mass | x̄ = (1/M)∭_D x·ρ dV |
| Moment of inertia | I_z = ∭_D (x² + y²)·ρ dV |

### Example 6

```
Find volume of tetrahedron bounded by x = 0, y = 0, z = 0, x + y + z = 1

Region: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1-x, 0 ≤ z ≤ 1-x-y

Volume = ∭_D 1 dV
       = ∫₀¹ ∫₀^{1-x} ∫₀^{1-x-y} dz dy dx
       = ∫₀¹ ∫₀^{1-x} (1-x-y) dy dx
       = ∫₀¹ [(1-x)y - y²/2]₀^{1-x} dx
       = ∫₀¹ (1-x)² - (1-x)²/2 dx
       = ∫₀¹ (1-x)²/2 dx
       = [(1-x)³/(-6)]₀¹ = 0 - (-1/6) = 1/6
```

---

## 5. Change of Variables in Multiple Integrals

### Jacobian Determinant

For transformation T: (u, v) → (x, y) where x = x(u,v), y = y(u,v):

```
        | ∂x/∂u  ∂x/∂v |
J = det |             | = ∂x/∂u · ∂y/∂v - ∂x/∂v · ∂y/∂u
        | ∂y/∂u  ∂y/∂v |
```

### Change of Variables Formula

```
∬_D f(x, y) dx dy = ∬_{D'} f(x(u,v), y(u,v)) · |J| du dv
```

### Polar Coordinates (review)

```
x = r·cos(θ), y = r·sin(θ)

        | cos(θ)  -r·sin(θ) |
J = det |                   | = r·cos²(θ) + r·sin²(θ) = r
        | sin(θ)   r·cos(θ) |

dA = |J| dr dθ = r dr dθ  ✓
```

### Cylindrical Coordinates

```
x = r·cos(θ)
y = r·sin(θ)
z = z

dV = r dr dθ dz
```

### Spherical Coordinates

```
x = ρ·sin(φ)·cos(θ)
y = ρ·sin(φ)·sin(θ)
z = ρ·cos(φ)

        | ∂(x,y,z)/∂(ρ,φ,θ) | = ρ²·sin(φ)

dV = ρ²·sin(φ) dρ dφ dθ
```

### Example 7

```
Evaluate ∬_D (x² + y²) dA where D is the unit disk

Using polar coordinates:
x² + y² = r²
dA = r dr dθ
Region: 0 ≤ r ≤ 1, 0 ≤ θ ≤ 2π

    2π  1              2π  1
∬ r² · r dr dθ = ∫ ∫ r³ dr dθ
    0   0              0   0
    2π
= ∫ [r⁴/4]₀¹ dθ = ∫ (1/4) dθ = (1/4)·2π = π/2
    0                  0
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, integrate, exp, sin, cos, sqrt, pi, polar_lift
from scipy import integrate as scipy_integrate
from mpl_toolkits.mplot3d import Axes3D

x, y, z, r, theta, phi, rho = symbols('x y z r theta phi rho')

# === Double Integrals ===

print("=" * 60)
print("DOUBLE INTEGRALS")
print("=" * 60)

# Example 1: Rectangle
f = x**2 + y**2
result = integrate(integrate(f, (y, 0, 2)), (x, 0, 1))
print(f"∬_R (x² + y²) dA over [0,1]×[0,2] = {result}")

# Example 2: General region
f = x * y
# Region: 0 ≤ x ≤ 1, x² ≤ y ≤ x
result = integrate(integrate(f, (y, x**2, x)), (x, 0, 1))
print(f"\n∬_D xy dA over region bounded by y=x and y=x² = {result}")

# === Polar Coordinates ===

print("\n" + "=" * 60)
print("POLAR COORDINATES")
print("=" * 60)

# Volume under z = 4 - x² - y² over disk x² + y² ≤ 4
f_polar = (4 - r**2) * r  # Include Jacobian
result = integrate(integrate(f_polar, (r, 0, 2)), (theta, 0, 2*pi))
print(f"Volume under z = 4 - r² over disk r ≤ 2: {result}")

# Gaussian integral
gaussian_2d = exp(-r**2) * r
result = integrate(integrate(gaussian_2d, (r, 0, oo)), (theta, 0, 2*pi))
print(f"\n2D Gaussian integral: {result}")
print(f"This equals π, so ∫e^(-x²)dx = √π")

# === Triple Integrals ===

print("\n" + "=" * 60)
print("TRIPLE INTEGRALS")
print("=" * 60)

# Volume of tetrahedron
f = 1
result = integrate(integrate(integrate(f, (z, 0, 1-x-y)), 
                             (y, 0, 1-x)), 
                   (x, 0, 1))
print(f"Volume of tetrahedron (x+y+z≤1): {result}")

# Mass with density
f = x**2 + y**2 + z**2
result = integrate(integrate(integrate(f, (z, 0, 1)), 
                             (y, 0, 1)), 
                   (x, 0, 1))
print(f"\n∭ (x² + y² + z²) dV over unit cube: {result}")

# === Numerical Double Integration ===

print("\n" + "=" * 60)
print("NUMERICAL DOUBLE INTEGRATION")
print("=" * 60)

def double_integral_numerical(f, x_range, y_range, n=100):
    """Compute double integral numerically"""
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    dx = (x_max - x_min) / n
    dy = (y_max - y_min) / n
    
    total = 0
    for i in range(n):
        for j in range(n):
            x_mid = x_min + (i + 0.5) * dx
            y_mid = y_min + (j + 0.5) * dy
            total += f(x_mid, y_mid) * dx * dy
    
    return total

# Test: ∬ (x² + y²) over [0,1]×[0,1]
f = lambda x, y: x**2 + y**2
numerical_result = double_integral_numerical(f, (0, 1), (0, 1), n=200)
exact = 2/3
print(f"∬ (x² + y²) over [0,1]×[0,1]:")
print(f"  Numerical: {numerical_result:.10f}")
print(f"  Exact: {exact:.10f}")
print(f"  Error: {abs(numerical_result - exact):.2e}")

# === Visualization of Double Integral ===

def visualize_double_integral(f, x_range, y_range, n=30):
    """Visualize volume under surface"""
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_vals = np.linspace(x_min, x_max, n)
    y_vals = np.linspace(y_min, y_max, n)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title(f'z = f(x, y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Wireframe with volume indication
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='blue', alpha=0.5)
    
    # Fill bottom
    ax2.plot_surface(X, Y, np.zeros_like(Z), alpha=0.3, color='green')
    
    ax2.set_title('Volume Representation')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    plt.tight_layout()
    plt.show()
    
    # Compute numerical integral
    result = double_integral_numerical(f, x_range, y_range, n=100)
    print(f"Volume under surface: {result:.6f}")

# Example
f = lambda x, y: 4 - x**2 - y**2
visualize_double_integral(f, (-2, 2), (-2, 2))

# === Polar Integration Visualization ===

def polar_double_integral(f_polar, r_range, theta_range, n_r=50, n_theta=50):
    """Compute double integral in polar coordinates"""
    r_min, r_max = r_range
    theta_min, theta_max = theta_range
    
    dr = (r_max - r_min) / n_r
    dtheta = (theta_max - theta_min) / n_theta
    
    total = 0
    for i in range(n_r):
        for j in range(n_theta):
            r_mid = r_min + (i + 0.5) * dr
            theta_mid = theta_min + (j + 0.5) * dtheta
            # Include Jacobian r
            total += f_polar(r_mid, theta_mid) * r_mid * dr * dtheta
    
    return total

# Example: ∬ r² over unit disk
f_polar = lambda r, theta: r**2
result = polar_double_integral(f_polar, (0, 1), (0, 2*np.pi))
exact = np.pi / 2
print(f"\n∬ r² over unit disk:")
print(f"  Numerical: {result:.10f}")
print(f"  Exact (π/2): {exact:.10f}")
print(f"  Error: {abs(result - exact):.2e}")

# === Change of Variables Demo ===

def demonstrate_change_of_variables():
    """Demonstrate change of variables in double integrals"""
    
    print("\n" + "=" * 60)
    print("CHANGE OF VARIABLES")
    print("=" * 60)
    
    # Original integral: ∬ (x² + y²) over unit disk
    # In polar: ∬ r² · r dr dθ
    
    # Compute in Cartesian (harder)
    def integrand_cartesian(y, x):
        return x**2 + y**2
    
    # Using scipy for 2D integration
    # Note: This requires careful handling of the circular boundary
    result_cartesian, error = scipy_integrate.dblquad(
        integrand_cartesian,
        -1, 1,  # x limits
        lambda x: -np.sqrt(1 - x**2),  # y lower
        lambda x: np.sqrt(1 - x**2)    # y upper
    )
    
    # Compute in polar (easier)
    def integrand_polar(theta, r):
        return r**2 * r  # Include Jacobian
    
    result_polar, error = scipy_integrate.dblquad(
        integrand_polar,
        0, 1,  # r limits
        0, 2*np.pi  # theta limits
    )
    
    print(f"∬ (x² + y²) over unit disk:")
    print(f"  Cartesian coordinates: {result_cartesian:.10f}")
    print(f"  Polar coordinates: {result_polar:.10f}")
    print(f"  Exact (π/2): {np.pi/2:.10f}")

demonstrate_change_of_variables()

# === Using scipy for complex integrals ===

print("\n" + "=" * 60)
print("SCIPY INTEGRATION")
print("=" * 60)

# Triple integral example
def integrand_3d(z, y, x):
    return x**2 + y**2 + z**2

result, error = scipy_integrate.tplquad(
    integrand_3d,
    0, 1,  # x limits
    0, 1,  # y limits
    0, 1   # z limits
)

print(f"∭ (x² + y² + z²) dV over unit cube:")
print(f"  Numerical: {result:.10f}")
print(f"  Exact: {1.0:.10f}")
print(f"  Error: {abs(result - 1):.2e}")
```

---

## 📊 Summary Table

| Concept | Formula | Notes |
|---------|---------|-------|
| **Double (rectangle)** | ∫ₐᵇ∫_c^d f(x,y) dy dx | Fubini's theorem |
| **Double (general)** | ∫ₐᵇ∫_{g₁(x)}^{g₂(x)} f(x,y) dy dx | Type I region |
| **Polar** | ∬ f(r,θ) r dr dθ | Jacobian = r |
| **Triple** | ∫∫∫ f(x,y,z) dz dy dx | Volume, mass |
| **Cylindrical** | ∭ f(r,θ,z) r dr dθ dz | For cylinders |
| **Spherical** | ∭ f(ρ,φ,θ) ρ²sin(φ) dρ dφ dθ | For spheres |
| **Change of variables** | ∬ f(x,y) dx dy = ∬ f(u,v) \|J\| du dv | Jacobian |

---

## 🎯 ML Applications

| Application | Integration Concept |
|-------------|-------------------|
| **Joint Probability** | P(X,Y) = ∬ f(x,y) dx dy |
| **Marginal Distribution** | P(X) = ∫ f(x,y) dy |
| **Expected Value** | E[X] = ∬ x·f(x,y) dx dy |
| **Covariance** | Cov(X,Y) = ∬ (x-μₓ)(y-μᵧ)f(x,y) dx dy |
| **Bayesian Inference** | Posterior ∝ Likelihood × Prior |
| **Normalizing Flows** | Change of variables formula |

---

## ❓ Quick Check Questions

1. What does a double integral represent geometrically?
2. State Fubini's theorem.
3. When should you use polar coordinates?
4. What is the Jacobian for polar coordinates?
5. How do you set up a triple integral for volume?
6. Why is change of variables useful?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Double integral geometric meaning:**
   - Volume under surface z = f(x, y)

2. **Fubini's theorem:**
   - For continuous functions on rectangles
   - Order of integration doesn't matter

3. **Use polar when:**
   - Circular regions
   - Functions with x² + y²

4. **Polar Jacobian:**
   - |J| = r
   - dA = r dr dθ

5. **Triple integral for volume:**
   - V = ∭_D 1 dV
   - Set up bounds based on region

6. **Change of variables useful:**
   - Simplifies region boundaries
   - Simplifies integrand
   - Makes integration possible

</details>
---

**Status:** ✅ Complete
**Next:** Vector Calculus
