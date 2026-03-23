# 1.2.8 Integration Fundamentals

## 🎯 Quick Overview
- **Integral**: Antiderivative, area under curve
- **Fundamental Theorem**: Connects differentiation and integration
- **Foundation for**: Probability (continuous distributions), expected values, loss functions

---

## 1. Antiderivatives and Indefinite Integrals

### Definition

An **antiderivative** of f(x) is a function F(x) such that:
```
F'(x) = f(x)
```

**Indefinite Integral:**
```
∫ f(x) dx = F(x) + C

where C is an arbitrary constant
```

### Why the Constant C?

```
If F(x) is an antiderivative of f(x), then:
(F(x) + C)' = F'(x) + 0 = f(x)

So ALL functions F(x) + C are antiderivatives
```

### Basic Integration Rules

| Function | Indefinite Integral |
|----------|-------------------|
| xⁿ (n ≠ -1) | xⁿ⁺¹/(n+1) + C |
| 1/x | ln\|x\| + C |
| eˣ | eˣ + C |
| aˣ | aˣ/ln(a) + C |
| sin(x) | -cos(x) + C |
| cos(x) | sin(x) + C |
| sec²(x) | tan(x) + C |
| 1/(1+x²) | arctan(x) + C |
| 1/√(1-x²) | arcsin(x) + C |

### Examples

**Example 1:**
```
∫ (3x² + 2x + 1) dx = x³ + x² + x + C
```

**Example 2:**
```
∫ (x³ - 2x² + 5) dx = x⁴/4 - 2x³/3 + 5x + C
```

**Example 3:**
```
∫ (eˣ + 2cos(x)) dx = eˣ + 2sin(x) + C
```

---

## 2. Definite Integrals and Area Under Curve

### Definition

The **definite integral** of f(x) from a to b:
```
    b
∫ f(x) dx = limit of Riemann sums
    a
```

### Riemann Sum

**Partition [a, b] into n subintervals:**
```
Δx = (b - a)/n
xᵢ = a + i·Δx

    b              n
∫ f(x) dx = lim Σ f(xᵢ*)·Δx
    a       n→∞  i=1

where xᵢ* is any point in [xᵢ₋₁, xᵢ]
```

### Geometric Interpretation

```
    b
∫ f(x) dx = Net area between curve and x-axis from a to b
    a

Areas above x-axis: positive
Areas below x-axis: negative
```

### Example

```
    2
∫ x² dx
    0

Using antiderivative F(x) = x³/3:

    2
∫ x² dx = [x³/3]₀² = 2³/3 - 0³/3 = 8/3
    0
```

---

## 3. Fundamental Theorem of Calculus

### Part 1 (Differentiation of Integrals)

**If f is continuous and:**
```
        x
F(x) = ∫ f(t) dt
        a

Then: F'(x) = f(x)
```

**Meaning:** The derivative of an integral gives back the original function.

### Part 2 (Evaluation Theorem)

**If F is any antiderivative of f:**
```
    b
∫ f(x) dx = F(b) - F(a) = [F(x)]ₐᵇ
    a
```

**Meaning:** Definite integrals can be computed using antiderivatives!

### Examples

**Example 4:**
```
    π
∫ sin(x) dx = [-cos(x)]₀^π = -cos(π) - (-cos(0)) = -(-1) + 1 = 2
    0
```

**Example 5:**
```
    3
∫ (2x + 1) dx = [x² + x]₁³ = (9 + 3) - (1 + 1) = 10
    1
```

**Example 6 (Using FTC Part 1):**
```
        x²
F(x) = ∫ eᵗ dt
        0

dF/dx = e^(x²) · 2x  (chain rule!)
```

---

## 4. Integration Techniques

### 4.1 Substitution (u-substitution)

**When:** Integral contains a function and its derivative

**Method:**
```
1. Let u = g(x) (choose inner function)
2. Compute du = g'(x) dx
3. Substitute: ∫ f(g(x))g'(x) dx = ∫ f(u) du
4. Integrate in terms of u
5. Substitute back: u = g(x)
```

**Examples:**

```
∫ 2x·e^(x²) dx

Let u = x², then du = 2x dx

∫ eᵘ du = eᵘ + C = e^(x²) + C
```

```
    1
∫ 2x·√(x² + 1) dx
    0

Let u = x² + 1, then du = 2x dx

When x = 0: u = 1
When x = 1: u = 2

    2                  2
∫ √u du = [2u^(3/2)/3] = (2/3)(2^(3/2) - 1) = (2/3)(2√2 - 1)
    1                  1
```

---

### 4.2 Integration by Parts

**Formula:**
```
∫ u dv = uv - ∫ v du

Or: ∫ f(x)g'(x) dx = f(x)g(x) - ∫ g(x)f'(x) dx
```

**LIATE Rule** (choose u in this order):
- **L**ogarithmic
- **I**nverse trigonometric
- **A**lgebraic
- **T**rigonometric
- **E**xponential

**Examples:**

```
∫ x·eˣ dx

Let u = x (algebraic), dv = eˣ dx
Then du = dx, v = eˣ

∫ x·eˣ dx = x·eˣ - ∫ eˣ dx = x·eˣ - eˣ + C = eˣ(x - 1) + C
```

```
∫ x²·sin(x) dx

Let u = x², dv = sin(x) dx
Then du = 2x dx, v = -cos(x)

∫ x²·sin(x) dx = -x²·cos(x) - ∫ (-cos(x))·2x dx
               = -x²·cos(x) + 2∫ x·cos(x) dx

Now integrate ∫ x·cos(x) dx by parts again:
Let u = x, dv = cos(x) dx
Then du = dx, v = sin(x)

∫ x·cos(x) dx = x·sin(x) - ∫ sin(x) dx = x·sin(x) + cos(x)

Final answer:
∫ x²·sin(x) dx = -x²·cos(x) + 2(x·sin(x) + cos(x)) + C
               = -x²·cos(x) + 2x·sin(x) + 2cos(x) + C
```

---

### 4.3 Trigonometric Substitution

**When:** Integral contains √(a² - x²), √(a² + x²), or √(x² - a²)

| Expression | Substitution | Identity |
|------------|--------------|----------|
| √(a² - x²) | x = a·sin(θ) | 1 - sin²(θ) = cos²(θ) |
| √(a² + x²) | x = a·tan(θ) | 1 + tan²(θ) = sec²(θ) |
| √(x² - a²) | x = a·sec(θ) | sec²(θ) - 1 = tan²(θ) |

**Example:**
```
∫ √(1 - x²) dx

Let x = sin(θ), dx = cos(θ) dθ
√(1 - x²) = √(1 - sin²(θ)) = cos(θ)

∫ cos(θ) · cos(θ) dθ = ∫ cos²(θ) dθ

Using cos²(θ) = (1 + cos(2θ))/2:

∫ (1 + cos(2θ))/2 dθ = θ/2 + sin(2θ)/4 + C

Back-substitute:
θ = arcsin(x)
sin(2θ) = 2sin(θ)cos(θ) = 2x√(1-x²)

Final: ∫ √(1 - x²) dx = arcsin(x)/2 + x√(1-x²)/2 + C
```

---

### 4.4 Partial Fraction Decomposition

**When:** Integrating rational functions P(x)/Q(x)

**Steps:**
1. Ensure degree(P) < degree(Q) (divide if needed)
2. Factor Q(x) completely
3. Decompose into simpler fractions
4. Solve for coefficients
5. Integrate each term

**Cases:**

**Case 1: Distinct linear factors**
```
        1             A     B
────────────── = ──────── + ────────
(x - 1)(x - 2)   (x - 1)   (x - 2)

1 = A(x - 2) + B(x - 1)

Let x = 1: 1 = A(-1) → A = -1
Let x = 2: 1 = B(1) → B = 1

        1            -1      1
∫ ────────────── dx = ∫ ───── dx + ∫ ───── dx
  (x - 1)(x - 2)      (x - 1)   (x - 2)

= -ln|x - 1| + ln|x - 2| + C
= ln|(x - 2)/(x - 1)| + C
```

**Case 2: Repeated linear factors**
```
        1             A     B
────────────── = ──────── + ────────
(x - 1)²         (x - 1)   (x - 1)²
```

**Case 3: Irreducible quadratic factors**
```
        1           Ax + B
────────────── = ──────────
(x² + 1)(x - 1)   (x² + 1)(x - 1)
```

---

## 5. Improper Integrals

### Type 1: Infinite Intervals

```
    ∞                b
∫ f(x) dx = lim ∫ f(x) dx
    a        b→∞ a

    ∞              t
∫ f(x) dx = lim ∫ f(x) dx
   -∞       t→-∞ -∞

    ∞               0              ∞
∫ f(x) dx = ∫ f(x) dx + ∫ f(x) dx
   -∞            -∞             0
```

**Examples:**

```
    ∞
∫ e^(-x) dx = lim [-e^(-x)]₁ᵗ = lim (-e^(-t) + e^(-1)) = 1/e
    1       t→∞                    t→∞

Converges to 1/e
```

```
    ∞
∫ 1/x dx = lim [ln|x|]₁ᵗ = lim ln(t) = ∞
    1       t→∞          t→∞

Diverges
```

### Type 2: Discontinuous Integrand

```
    b              b
∫ f(x) dx = lim ∫ f(x) dx
    a       t→a⁺ t

    b              t
∫ f(x) dx = lim ∫ f(x) dx
    a       t→b⁻ a
```

**Example:**
```
    1
∫ 1/√x dx = lim [2√x]ᵗ₁ = lim (2 - 2√t) = 2
    0       t→0⁺           t→0⁺

Converges to 2
```

---

## 6. Numerical Integration

### Riemann Sums

**Left endpoint:**
```
    b          n-1
∫ f(x) dx ≈ Σ f(xᵢ)·Δx
    a         i=0
```

**Right endpoint:**
```
    b          n
∫ f(x) dx ≈ Σ f(xᵢ)·Δx
    a         i=1
```

### Trapezoidal Rule

```
    b                    Δx
∫ f(x) dx ≈ ──── [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]
    a                     2

where Δx = (b-a)/n, xᵢ = a + i·Δx
```

### Simpson's Rule (n must be even)

```
    b                    Δx
∫ f(x) dx ≈ ──── [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]
    a                     3
```

**Error Comparison:**
| Method | Error Order |
|--------|-------------|
| Riemann | O(1/n) |
| Trapezoidal | O(1/n²) |
| Simpson's | O(1/n⁴) |

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, integrate, exp, sin, cos, ln, sqrt, oo, limit

x = symbols('x')

# === Symbolic Integration ===

print("=" * 60)
print("SYMBOLIC INTEGRATION")
print("=" * 60)

# Indefinite integrals
f = x**2 + 2*x + 1
indefinite = integrate(f, x)
print(f"∫ ({f}) dx = {indefinite} + C")

f = exp(x) * cos(x)
indefinite = integrate(f, x)
print(f"\n∫ eˣ·cos(x) dx = {indefinite}")

# Definite integrals
f = x**2
definite = integrate(f, (x, 0, 2))
print(f"\n∫₀² x² dx = {definite}")

f = sin(x)
definite = integrate(f, (x, 0, np.pi))
print(f"∫₀^π sin(x) dx = {definite}")

# === Integration Techniques ===

print("\n" + "=" * 60)
print("INTEGRATION TECHNIQUES")
print("=" * 60)

# Substitution example
print("\nSubstitution: ∫ 2x·e^(x²) dx")
u = x**2
f_sub = exp(u)
result_sub = integrate(f_sub, u)
print(f"Let u = x², du = 2x dx")
print(f"∫ eᵘ du = {result_sub} + C = {result_sub.subs(u, x**2)} + C")

# Integration by parts
print("\nIntegration by parts: ∫ x·eˣ dx")
f_parts = x * exp(x)
result_parts = integrate(f_parts, x)
print(f"∫ x·eˣ dx = {result_parts}")

# Partial fractions
print("\nPartial fractions: ∫ 1/(x² - 1) dx")
f_pf = 1 / (x**2 - 1)
result_pf = integrate(f_pf, x)
print(f"∫ 1/(x² - 1) dx = {result_pf}")

# === Improper Integrals ===

print("\n" + "=" * 60)
print("IMPROPER INTEGRALS")
print("=" * 60)

# Convergent improper integral
f_improper = exp(-x)
result_improper = integrate(f_improper, (x, 1, oo))
print(f"∫₁^∞ e^(-x) dx = {result_improper}")

# Divergent improper integral
f_div = 1/x
result_div = integrate(f_div, (x, 1, oo))
print(f"∫₁^∞ 1/x dx = {result_div}")

# === Numerical Integration ===

print("\n" + "=" * 60)
print("NUMERICAL INTEGRATION")
print("=" * 60)

def riemann_left(f, a, b, n):
    """Left Riemann sum"""
    dx = (b - a) / n
    x_vals = np.linspace(a, b - dx, n)
    return np.sum(f(x_vals)) * dx

def riemann_right(f, a, b, n):
    """Right Riemann sum"""
    dx = (b - a) / n
    x_vals = np.linspace(a + dx, b, n)
    return np.sum(f(x_vals)) * dx

def trapezoidal(f, a, b, n):
    """Trapezoidal rule"""
    dx = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    y_vals = f(x_vals)
    return dx * (0.5 * y_vals[0] + np.sum(y_vals[1:-1]) + 0.5 * y_vals[-1])

def simpsons(f, a, b, n):
    """Simpson's rule (n must be even)"""
    if n % 2 == 1:
        n += 1
    dx = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    y_vals = f(x_vals)
    
    coeffs = np.ones(n + 1)
    coeffs[1:-1:2] = 4
    coeffs[2:-2:2] = 2
    
    return (dx / 3) * np.sum(coeffs * y_vals)

# Test on ∫₀¹ x² dx = 1/3
f = lambda x: x**2
a, b = 0, 1
exact = 1/3

print(f"\nIntegrating f(x) = x² from {a} to {b}")
print(f"Exact value: {exact}")
print(f"\n{'n':<6} {'Left':<12} {'Right':<12} {'Trap':<12} {'Simp':<12}")
print("-" * 55)

for n in [4, 8, 16, 32, 64]:
    left = riemann_left(f, a, b, n)
    right = riemann_right(f, a, b, n)
    trap = trapezoidal(f, a, b, n)
    simp = simpsons(f, a, b, n)
    
    print(f"{n:<6} {left:<12.8f} {right:<12.8f} {trap:<12.8f} {simp:<12.8f}")

# === Visualization of Integration ===

def visualize_riemann(f, a, b, n, method='left'):
    """Visualize Riemann sum approximation"""
    dx = (b - a) / n
    x_vals = np.linspace(a, b, 400)
    y_vals = f(x_vals)
    
    if method == 'left':
        x_rect = np.linspace(a, b - dx, n)
    elif method == 'right':
        x_rect = np.linspace(a + dx, b, n)
    else:
        x_rect = np.linspace(a + dx/2, b - dx/2, n)  # midpoint
    
    y_rect = f(x_rect)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Function with rectangles
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    
    for i, x_r in enumerate(x_rect):
        rect = plt.Rectangle((x_r, 0), dx, y_rect[i], 
                            alpha=0.3, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)
    
    plt.title(f'{method.capitalize()} Riemann Sum (n={n})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Convergence
    plt.subplot(1, 2, 2)
    n_values = [4, 8, 16, 32, 64, 128]
    approximations = []
    
    for n_val in n_values:
        if method == 'left':
            approx = riemann_left(f, a, b, n_val)
        else:
            approx = riemann_right(f, a, b, n_val)
        approximations.append(approx)
    
    plt.plot(n_values, approximations, 'bo-', linewidth=2, label='Approximation')
    plt.axhline(y=exact, color='r', linestyle='--', label='Exact')
    plt.title('Convergence as n increases')
    plt.xlabel('n')
    plt.ylabel('Integral value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example
f = lambda x: x**2
visualize_riemann(f, 0, 1, 8, method='left')

# === Area Between Curves ===

def area_between_curves(f, g, a, b, n_points=400):
    """Calculate and visualize area between two curves"""
    x_vals = np.linspace(a, b, n_points)
    y_f = f(x_vals)
    y_g = g(x_vals)
    
    # Area = ∫|f(x) - g(x)| dx
    area = np.trapz(np.abs(y_f - y_g), x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_f, 'b-', linewidth=2, label='f(x)')
    plt.plot(x_vals, y_g, 'r-', linewidth=2, label='g(x)')
    plt.fill_between(x_vals, y_f, y_g, alpha=0.3, label=f'Area = {area:.4f}')
    plt.title('Area Between Curves')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return area

# Example: Area between y = x² and y = x
f = lambda x: x
g = lambda x: x**2
area = area_between_curves(f, g, 0, 1)
print(f"\nArea between y = x and y = x² from 0 to 1: {area:.6f}")
print(f"Exact: 1/6 = {1/6:.6f}")

# === Arc Length ===

def arc_length(f, a, b, n=1000):
    """Compute arc length numerically"""
    from scipy.misc import derivative
    
    x_vals = np.linspace(a, b, n)
    dx = (b - a) / n
    
    # Arc length = ∫ √(1 + (f'(x))²) dx
    f_prime_vals = np.array([derivative(f, x, dx=1e-6) for x in x_vals])
    integrand = np.sqrt(1 + f_prime_vals**2)
    
    return np.trapz(integrand, x_vals)

# Example: Arc length of y = x² from 0 to 1
f = lambda x: x**2
length = arc_length(f, 0, 1)
print(f"\nArc length of y = x² from 0 to 1: {length:.6f}")
```

---

## 📊 Summary Table

| Concept | Formula | Notes |
|---------|---------|-------|
| **Indefinite** | ∫ f(x) dx = F(x) + C | Family of functions |
| **Definite** | ∫ₐᵇ f(x) dx = F(b) - F(a) | Number (net area) |
| **FTC Part 1** | d/dx ∫ₐˣ f(t) dt = f(x) | Derivative of integral |
| **FTC Part 2** | ∫ₐᵇ f(x) dx = F(b) - F(a) | Evaluation theorem |
| **Substitution** | ∫ f(g(x))g'(x) dx = ∫ f(u) du | Reverse chain rule |
| **By Parts** | ∫ u dv = uv - ∫ v du | Reverse product rule |
| **Trapezoidal** | (Δx/2)[f(x₀) + 2Σf(xᵢ) + f(xₙ)] | O(1/n²) error |
| **Simpson's** | (Δx/3)[f(x₀) + 4Σodd + 2Σeven + f(xₙ)] | O(1/n⁴) error |

---

## 🎯 ML Applications

| Application | Integration Concept |
|-------------|-------------------|
| **Probability** | P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx |
| **Expected Value** | E[X] = ∫ x·f(x) dx |
| **Cross-entropy Loss** | Involves integrals over distributions |
| **Bayesian Inference** | Marginalization via integration |
| **Normalizing Flows** | Change of variables formula |

---

## ❓ Quick Check Questions

1. What's the difference between definite and indefinite integrals?
2. State the Fundamental Theorem of Calculus (both parts).
3. When do you use substitution vs integration by parts?
4. What is an improper integral?
5. Compare accuracy of numerical integration methods.
6. How do you find area between curves?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Definite vs Indefinite:**
   - Indefinite: Family of functions (+ C)
   - Definite: Number (net area)

2. **FTC:**
   - Part 1: d/dx ∫ₐˣ f(t) dt = f(x)
   - Part 2: ∫ₐᵇ f(x) dx = F(b) - F(a)

3. **Substitution vs By Parts:**
   - Substitution: function + its derivative present
   - By parts: product of different function types

4. **Improper integral:**
   - Infinite interval OR discontinuous integrand
   - Evaluated using limits

5. **Numerical accuracy:**
   - Simpson's > Trapezoidal > Riemann
   - O(1/n⁴) vs O(1/n²) vs O(1/n)

6. **Area between curves:**
   - ∫ |f(x) - g(x)| dx
   - Or ∫ (top - bottom) dx

</details>
---

**Status:** ✅ Complete
**Next:** Multiple Integration
