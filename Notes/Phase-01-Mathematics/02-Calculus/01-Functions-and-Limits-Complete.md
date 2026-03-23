# 1.2.1 Functions and Limits

## 🎯 Quick Overview
- **Function**: Mapping from inputs to outputs
- **Limit**: Value a function approaches as input approaches a point
- **Foundation for**: Derivatives, integrals, optimization, convergence analysis

---

## 1. Function Types

### Elementary Functions

| Type | Form | Example | Domain | Range |
|------|------|---------|--------|-------|
| **Polynomial** | aₙxⁿ + ... + a₁x + a₀ | f(x) = 2x³ - x + 5 | ℝ | ℝ |
| **Exponential** | aˣ or eˣ | f(x) = eˣ | ℝ | (0, ∞) |
| **Logarithmic** | logₐ(x) or ln(x) | f(x) = ln(x) | (0, ∞) | ℝ |
| **Trigonometric** | sin(x), cos(x), tan(x) | f(x) = sin(x) | ℝ | [-1, 1] |
| **Rational** | P(x)/Q(x) | f(x) = 1/x | ℝ\{0} | ℝ\{0} |
| **Power** | xⁿ | f(x) = x² | ℝ | [0, ∞) |

### Key Properties

**Exponential Function (eˣ):**
- eˣ > 0 for all x
- e⁰ = 1
- eˣ⁺ʸ = eˣ · eʸ
- d/dx(eˣ) = eˣ (unique property!)

**Natural Logarithm (ln x):**
- ln(x) defined only for x > 0
- ln(1) = 0
- ln(e) = 1
- ln(xy) = ln(x) + ln(y)
- d/dx(ln x) = 1/x

**Trigonometric Functions:**
```
sin²(x) + cos²(x) = 1
sin(-x) = -sin(x)  (odd function)
cos(-x) = cos(x)   (even function)
```

---

## 2. Composite and Inverse Functions

### Composite Functions

**Definition:** (f ∘ g)(x) = f(g(x))

**Example:**
```
f(x) = x², g(x) = x + 1

(f ∘ g)(x) = f(g(x)) = f(x+1) = (x+1)²
(g ∘ f)(x) = g(f(x)) = g(x²) = x² + 1

Note: (f ∘ g)(x) ≠ (g ∘ f)(x) in general!
```

### Inverse Functions

**Definition:** f⁻¹ is the inverse of f if:
```
f(f⁻¹(x)) = x  and  f⁻¹(f(x)) = x
```

**Requirements:**
- Function must be **one-to-one** (passes horizontal line test)

**Common Inverses:**
| Function | Inverse | Domain Restriction |
|----------|---------|-------------------|
| eˣ | ln(x) | x > 0 |
| sin(x) | arcsin(x) | -π/2 ≤ x ≤ π/2 |
| cos(x) | arccos(x) | 0 ≤ x ≤ π |
| x² | √x | x ≥ 0 |

---

## 3. Limit Definition and Intuition

### Informal Definition

The **limit** of f(x) as x approaches a is L:
```
lim(x→a) f(x) = L
```
Meaning: f(x) gets arbitrarily close to L as x gets sufficiently close to a (but x ≠ a)

### Formal (ε-δ) Definition

```
lim(x→a) f(x) = L

means:

For every ε > 0, there exists δ > 0 such that:
if 0 < |x - a| < δ, then |f(x) - L| < ε
```

**Visual Interpretation:**
```
ε (epsilon) = tolerance for output
δ (delta) = tolerance for input

No matter how small ε, we can find δ that works
```

### Examples

**Example 1:** lim(x→2) (3x + 1) = 7
```
As x → 2, 3x + 1 → 3(2) + 1 = 7
```

**Example 2:** lim(x→0) sin(x)/x = 1
```
This is a FUNDAMENTAL limit in calculus
Used to prove d/dx(sin x) = cos x
```

**Example 3:** lim(x→∞) 1/x = 0
```
As x grows without bound, 1/x approaches 0
```

---

## 4. One-Sided Limits

### Left-Hand Limit
```
lim(x→a⁻) f(x) = L

x approaches a from values LESS than a
```

### Right-Hand Limit
```
lim(x→a⁺) f(x) = L

x approaches a from values GREATER than a
```

### Two-Sided Limit Exists IFF:
```
lim(x→a⁻) f(x) = lim(x→a⁺) f(x) = L

Then: lim(x→a) f(x) = L
```

**Example:**
```
        { x + 1,  x < 2
f(x) = {
        { x²,     x ≥ 2

lim(x→2⁻) f(x) = 2 + 1 = 3
lim(x→2⁺) f(x) = 2² = 4

Since 3 ≠ 4, lim(x→2) f(x) DOES NOT EXIST
```

---

## 5. Infinite Limits and Limits at Infinity

### Infinite Limits

**Vertical Asymptote:**
```
lim(x→a) f(x) = ∞  or  lim(x→a) f(x) = -∞

Example: lim(x→0) 1/x² = ∞
```

### Limits at Infinity

**Horizontal Asymptote:**
```
lim(x→∞) f(x) = L  or  lim(x→-∞) f(x) = L

Example: lim(x→∞) 1/x = 0
```

### Techniques for Limits at Infinity

**Rational Functions:**
```
        3x² + 2x + 1
lim(x→∞) ────────────
          2x² - x + 5

Divide by highest power:

        3 + 2/x + 1/x²       3 + 0 + 0    3
= lim(x→∞) ───────────── = ─────────── = ─
          2 - 1/x + 5/x²      2 - 0 + 0    2
```

**General Rules:**
| Degrees | Limit |
|---------|-------|
| deg(numerator) < deg(denominator) | 0 |
| deg(numerator) = deg(denominator) | ratio of leading coefficients |
| deg(numerator) > deg(denominator) | ∞ or -∞ |

---

## 6. Continuity

### Definition

f is **continuous** at x = a if:
1. f(a) is defined
2. lim(x→a) f(x) exists
3. lim(x→a) f(x) = f(a)

### Types of Discontinuity

| Type | Description | Example |
|------|-------------|---------|
| **Removable** | Limit exists, but f(a) undefined or ≠ limit | f(x) = (x²-1)/(x-1) at x=1 |
| **Jump** | Left and right limits exist but differ | Step function |
| **Infinite** | One or both one-sided limits are ±∞ | f(x) = 1/x at x=0 |
| **Oscillating** | Function oscillates without settling | f(x) = sin(1/x) at x=0 |

### Continuous Functions

**Always Continuous on Domain:**
- Polynomials (everywhere)
- Rational functions (where denominator ≠ 0)
- Trigonometric functions
- Exponential functions
- Logarithmic functions
- Compositions of continuous functions

---

## 7. Important Theorems

### Intermediate Value Theorem (IVT)

**Statement:**
```
If f is continuous on [a, b] and N is any value between f(a) and f(b),
then there exists c ∈ (a, b) such that f(c) = N
```

**Application:**
```
Prove x³ - x - 1 = 0 has a root in [1, 2]:

f(1) = 1 - 1 - 1 = -1
f(2) = 8 - 2 - 1 = 5

Since f(1) < 0 < f(2) and f is continuous,
by IVT, there exists c ∈ (1, 2) where f(c) = 0
```

### Squeeze Theorem (Sandwich Theorem)

**Statement:**
```
If g(x) ≤ f(x) ≤ h(x) for all x near a,
and lim(x→a) g(x) = lim(x→a) h(x) = L,
then lim(x→a) f(x) = L
```

**Classic Example:**
```
Prove: lim(x→0) x²·sin(1/x) = 0

Since -1 ≤ sin(1/x) ≤ 1:
-x² ≤ x²·sin(1/x) ≤ x²

lim(x→0) -x² = 0 and lim(x→0) x² = 0

By Squeeze Theorem: lim(x→0) x²·sin(1/x) = 0
```

---

## 8. Asymptotes

### Vertical Asymptotes

**Find where denominator = 0:**
```
        x + 1
f(x) = ───────
        x - 2

Vertical asymptote: x = 2
```

### Horizontal Asymptotes

**Evaluate limit at infinity:**
```
        2x + 1
f(x) = ───────
        x - 3

lim(x→∞) f(x) = 2

Horizontal asymptote: y = 2
```

### Oblique (Slant) Asymptotes

**When degree(numerator) = degree(denominator) + 1:**
```
        x² + 1
f(x) = ───────
         x

Oblique asymptote: y = x (from polynomial division)
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, limit, oo, sin, cos, exp, ln, plot

x = symbols('x')

# === Limit Calculations ===

# Basic limit
print(f"lim(x→2) 3x + 1 = {limit(3*x + 1, x, 2)}")

# Limit at infinity
print(f"lim(x→∞) 1/x = {limit(1/x, x, oo)}")

# Famous limit
print(f"lim(x→0) sin(x)/x = {limit(sin(x)/x, x, 0)}")

# One-sided limits
print(f"lim(x→0⁺) 1/x = {limit(1/x, x, 0, dir='+')}")
print(f"lim(x→0⁻) 1/x = {limit(1/x, x, 0, dir='-')}")

# === Continuity Check ===
def check_continuity(f, a):
    """Check if function is continuous at point a"""
    try:
        f_at_a = f.subs(x, a)
        lim_at_a = limit(f, x, a)
        is_continuous = f_at_a == lim_at_a
        return is_continuous
    except:
        return False

# === Visualization ===

# Plot function with asymptote
f = 1 / (x - 2)
plot(f, (x, -5, 5), ylim=(-10, 10), 
     title='f(x) = 1/(x-2) - Vertical Asymptote at x=2',
     line_color='blue')

# Plot to show limit concept
x_vals = np.linspace(-0.5, 0.5, 1000)
y_vals = np.sin(x_vals) / x_vals

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.axhline(y=1, color='r', linestyle='--', label='Limit = 1')
plt.axvline(x=0, color='g', linestyle=':', label='x = 0')
plt.title('lim(x→0) sin(x)/x = 1')
plt.xlabel('x')
plt.ylabel('sin(x)/x')
plt.legend()
plt.grid(True)
plt.show()

# === Numerical Limit Approximation ===
def approximate_limit(f, a, n_points=10):
    """Approximate limit numerically"""
    h_values = [10**(-i) for i in range(1, n_points+1)]
    
    print(f"\nApproximating lim(x→{a}) f(x):")
    print(f"{'h':<10} {'a-h':<15} {'a+h':<15} {'Average'}")
    print("-" * 55)
    
    for h in h_values:
        left = f(a - h)
        right = f(a + h)
        avg = (left + right) / 2
        print(f"{h:<10.2e} {left:<15.10f} {right:<15.10f} {avg:.10f}")

# Example: lim(x→0) sin(x)/x
approximate_limit(lambda x: np.sin(x)/x if x != 0 else 1, 0)
```

---

## 📊 Summary Table

| Concept | Notation | Key Idea |
|---------|----------|----------|
| **Limit** | lim(x→a) f(x) | Value approached |
| **Left Limit** | lim(x→a⁻) f(x) | Approach from left |
| **Right Limit** | lim(x→a⁺) f(x) | Approach from right |
| **Infinity Limit** | lim(x→a) f(x) = ∞ | Grows without bound |
| **Limit at Infinity** | lim(x→∞) f(x) | Behavior as x→∞ |
| **Continuity** | lim(x→a) f(x) = f(a) | No breaks/jumps |
| **IVT** | - | Root existence |
| **Squeeze Theorem** | - | Bounding technique |

---

## 🎯 ML Applications

| Application | Calculus Concept |
|-------------|-----------------|
| **Gradient Descent** | Limits define derivatives |
| **Loss Functions** | Continuity ensures smooth optimization |
| **Activation Functions** | Differentiability for backprop |
| **Convergence Analysis** | Limits at infinity |
| **Numerical Stability** | Avoiding discontinuities |

---

## ❓ Quick Check Questions

1. What is the difference between lim(x→a) f(x) and f(a)?
2. When does a two-sided limit NOT exist?
3. What are the three conditions for continuity?
4. How do you find horizontal asymptotes?
5. State the Intermediate Value Theorem in your own words.
6. Why is lim(x→0) sin(x)/x = 1 important?

---

## 📝 Answers to Quick Check

1. **Limit vs Function Value:**
   - lim(x→a) f(x) is the value f(x) approaches
   - f(a) is the actual value at x = a
   - They can differ (or f(a) might not exist)

2. **Two-sided limit doesn't exist when:**
   - Left and right limits are different
   - Function oscillates infinitely
   - Function goes to ±∞

3. **Continuity conditions:**
   - f(a) is defined
   - lim(x→a) f(x) exists
   - lim(x→a) f(x) = f(a)

4. **Finding horizontal asymptotes:**
   - Evaluate lim(x→∞) f(x) and lim(x→-∞) f(x)
   - For rational functions, compare degrees

5. **IVT:**
   - If f is continuous on [a,b] and N is between f(a) and f(b),
   - Then f must equal N at some point in (a,b)

6. **sin(x)/x limit importance:**
   - Used to prove d/dx(sin x) = cos x
   - Fundamental in signal processing (sinc function)
   - Appears in Fourier analysis

---

**Status:** ✅ Complete
**Next:** Differentiation Fundamentals
