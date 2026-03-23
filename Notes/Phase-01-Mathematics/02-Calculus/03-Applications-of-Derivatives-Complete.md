# 1.2.3 Applications of Derivatives

## 🎯 Quick Overview
- **Critical points**: Where f'(x) = 0 or undefined
- **Optimization**: Finding maximum/minimum values
- **Foundation for**: ML loss minimization, hyperparameter tuning

---

## 1. Critical Points and Stationary Points

### Definitions

**Critical Point:** x = c is a critical point if:
- f'(c) = 0, OR
- f'(c) does not exist

**Stationary Point:** x = c where f'(c) = 0
- All stationary points are critical points
- Not all critical points are stationary (could be where f' doesn't exist)

### Types of Critical Points

| Type | Description | Visual |
|------|-------------|--------|
| **Local Maximum** | f(c) ≥ f(x) for nearby x | Peak (∩) |
| **Local Minimum** | f(c) ≤ f(x) for nearby x | Valley (∪) |
| **Saddle/Inflection** | Neither max nor min | Flat but continuing |

### Examples

**Example 1:** f(x) = x³ - 3x² + 2

```
f'(x) = 3x² - 6x = 3x(x - 2)

Critical points: x = 0, x = 2

f(0) = 2  (local maximum)
f(2) = -2 (local minimum)
```

**Example 2:** f(x) = |x|

```
f'(x) = { -1,  x < 0
        {  1,  x > 0
        {  undefined, x = 0

Critical point: x = 0 (derivative doesn't exist)
This is a local (and global) minimum
```

---

## 2. First Derivative Test

### Theorem

If c is a critical point of f:

| Sign Change of f' | Conclusion |
|-------------------|------------|
| + to - | Local Maximum |
| - to + | Local Minimum |
| No change | Neither (inflection point) |

### Method

**Step 1:** Find f'(x)
**Step 2:** Find critical points (f'(x) = 0 or undefined)
**Step 3:** Test sign of f' in intervals around critical points
**Step 4:** Apply the table above

### Example

```
f(x) = x³ - 6x² + 9x + 1

Step 1: f'(x) = 3x² - 12x + 9 = 3(x² - 4x + 3) = 3(x-1)(x-3)

Step 2: Critical points: x = 1, x = 3

Step 3: Test intervals:
  x < 1:    f'(0) = 3(-1)(-3) = 9 > 0  (increasing)
  1 < x < 3: f'(2) = 3(1)(-1) = -3 < 0 (decreasing)
  x > 3:    f'(4) = 3(3)(1) = 9 > 0   (increasing)

Step 4: Conclusion:
  x = 1: + to - → Local Maximum, f(1) = 5
  x = 3: - to + → Local Minimum, f(3) = 1
```

---

## 3. Second Derivative Test

### Theorem

If f'(c) = 0 and f'' exists:

| f''(c) | Conclusion |
|--------|------------|
| f''(c) > 0 | Local Minimum (concave up) |
| f''(c) < 0 | Local Maximum (concave down) |
| f''(c) = 0 | Test inconclusive |

### Why It Works

```
f''(c) > 0 → concave up (∪) → minimum
f''(c) < 0 → concave down (∩) → maximum
```

### Example

```
f(x) = x⁴ - 4x³

f'(x) = 4x³ - 12x² = 4x²(x - 3)
f''(x) = 12x² - 24x = 12x(x - 2)

Critical points: x = 0, x = 3

Second derivative test:
f''(0) = 0 → Test inconclusive!
f''(3) = 12(3)(1) = 36 > 0 → Local Minimum

For x = 0, use first derivative test:
  x < 0: f' < 0 (decreasing)
  x > 0: f' < 0 (decreasing)
  No sign change → Inflection point, not extremum
```

---

## 4. Concavity and Inflection Points

### Definitions

**Concave Up (∪):**
- f''(x) > 0
- Tangent lines lie below curve
- Shape: cup (holds water)

**Concave Down (∩):**
- f''(x) < 0
- Tangent lines lie above curve
- Shape: cap (spills water)

**Inflection Point:**
- Point where concavity changes
- f''(x) = 0 AND sign of f'' changes

### Example

```
f(x) = x³ - 3x² + 2

f'(x) = 3x² - 6x
f''(x) = 6x - 6 = 6(x - 1)

f''(x) = 0 when x = 1

Test concavity:
  x < 1: f''(0) = -6 < 0 → Concave Down
  x > 1: f''(2) = 6 > 0 → Concave Up

Inflection point at x = 1
f(1) = 1 - 3 + 2 = 0
Inflection point: (1, 0)
```

---

## 5. Curve Sketching

### Complete Analysis Checklist

1. **Domain:** Where is f defined?
2. **Intercepts:** x-intercepts (f(x)=0), y-intercept (f(0))
3. **Symmetry:** Even f(-x)=f(x), Odd f(-x)=-f(x)
4. **Asymptotes:** Vertical, horizontal, slant
5. **Critical points:** f'(x) = 0 or undefined
6. **Increasing/Decreasing:** Sign of f'
7. **Local extrema:** Max/min values
8. **Concavity:** Sign of f''
9. **Inflection points:** Where f'' changes sign

### Example: Complete Curve Sketch

```
f(x) = x³/(x² - 1)

1. Domain: x ≠ ±1 (denominator ≠ 0)

2. Intercepts:
   f(0) = 0 → y-intercept at (0,0)
   f(x) = 0 → x = 0 → x-intercept at (0,0)

3. Symmetry:
   f(-x) = -x³/(x²-1) = -f(x) → Odd function (symmetric about origin)

4. Asymptotes:
   Vertical: x = 1, x = -1
   Horizontal: lim(x→∞) f(x) = ∞ (no horizontal)
   Slant: Do polynomial division

5. f'(x) = [3x²(x²-1) - x³(2x)]/(x²-1)² = [x⁴ - 3x²]/(x²-1)²
   Critical points: x = 0, x = ±√3

6. Sign analysis of f':
   ... (complete the analysis)

7-9. Continue with extrema, concavity, inflection points
```

---

## 6. Optimization Problems

### General Strategy

**Step 1:** Understand the problem, draw a diagram
**Step 2:** Define variables
**Step 3:** Write the objective function (what to optimize)
**Step 4:** Write constraints (equations relating variables)
**Step 5:** Express objective as function of ONE variable
**Step 6:** Find critical points
**Step 7:** Verify maximum/minimum (use second derivative or endpoints)
**Step 8:** Answer the original question

### Example 1: Rectangle with Maximum Area

```
Problem: Find the rectangle of maximum area with perimeter 20.

Step 1-2: Let width = w, height = h

Step 3: Maximize A = w·h

Step 4: Constraint: 2w + 2h = 20 → w + h = 10 → h = 10 - w

Step 5: A(w) = w(10-w) = 10w - w²

Step 6: A'(w) = 10 - 2w = 0 → w = 5

Step 7: A''(w) = -2 < 0 → Maximum

Step 8: w = 5, h = 5, Maximum Area = 25

Answer: Square with side 5 has maximum area
```

### Example 2: Closest Point on Curve

```
Problem: Find the point on y = x² closest to (0, 1).

Step 1-2: Point on curve is (x, x²)

Step 3: Minimize distance D = √[(x-0)² + (x²-1)²]

Easier: Minimize D² (same critical points)
f(x) = x² + (x²-1)² = x² + x⁴ - 2x² + 1 = x⁴ - x² + 1

Step 6: f'(x) = 4x³ - 2x = 2x(2x² - 1) = 0
  x = 0 or x = ±1/√2

Step 7: f''(x) = 12x² - 2
  f''(0) = -2 < 0 → Local max (not what we want)
  f''(±1/√2) = 12(1/2) - 2 = 4 > 0 → Local min ✓

Step 8: Points are (±1/√2, 1/2)
```

---

## 7. Related Rates

### Strategy

**Step 1:** Draw diagram, label variables
**Step 2:** Write equation relating variables
**Step 3:** Differentiate with respect to time t
**Step 4:** Substitute known values
**Step 5:** Solve for the unknown rate

### Example: Expanding Circle

```
Problem: Radius of a circle increases at 2 cm/s.
How fast is the area increasing when r = 5 cm?

Step 1: A = area, r = radius, dr/dt = 2 cm/s

Step 2: A = πr²

Step 3: Differentiate wrt t:
  dA/dt = 2πr · dr/dt

Step 4: When r = 5, dr/dt = 2:
  dA/dt = 2π(5)(2) = 20π

Step 5: Answer: Area increasing at 20π cm²/s ≈ 62.8 cm²/s
```

### Example: Sliding Ladder

```
Problem: 10-ft ladder slides down wall. Bottom slides away at 1 ft/s.
How fast is top sliding down when bottom is 6 ft from wall?

Step 1: x = distance from wall, y = height on wall
  dx/dt = 1 ft/s, find dy/dt when x = 6

Step 2: x² + y² = 100 (Pythagoras)

Step 3: 2x(dx/dt) + 2y(dy/dt) = 0
  x(dx/dt) + y(dy/dt) = 0

Step 4: When x = 6: y = √(100-36) = 8
  6(1) + 8(dy/dt) = 0

Step 5: dy/dt = -6/8 = -3/4 ft/s

Answer: Top is sliding down at 3/4 ft/s
```

---

## 8. Mean Value Theorem and Rolle's Theorem

### Rolle's Theorem

**Hypothesis:**
1. f is continuous on [a, b]
2. f is differentiable on (a, b)
3. f(a) = f(b)

**Conclusion:**
There exists c ∈ (a, b) such that f'(c) = 0

### Mean Value Theorem (MVT)

**Hypothesis:**
1. f is continuous on [a, b]
2. f is differentiable on (a, b)

**Conclusion:**
There exists c ∈ (a, b) such that:
```
f'(c) = [f(b) - f(a)] / (b - a)
```

**Geometric meaning:** Some tangent is parallel to secant line.

### Example

```
Verify MVT for f(x) = x² on [1, 4]:

[f(4) - f(1)] / (4 - 1) = (16 - 1) / 3 = 5

Find c where f'(c) = 5:
f'(x) = 2x
2c = 5 → c = 2.5

Check: 2.5 ∈ (1, 4) ✓
```

---

## 9. L'Hôpital's Rule

### Statement

If lim(x→a) f(x) = 0 and lim(x→a) g(x) = 0, OR
if lim(x→a) f(x) = ±∞ and lim(x→a) g(x) = ±∞:

```
lim(x→a) f(x)/g(x) = lim(x→a) f'(x)/g'(x)

(provided the limit on the right exists)
```

### Indeterminate Forms

| Form | Apply L'Hôpital? |
|------|-----------------|
| 0/0 | ✅ Yes |
| ∞/∞ | ✅ Yes |
| 0·∞ | Convert to 0/0 or ∞/∞ |
| ∞ - ∞ | Convert to quotient |
| 1^∞ | Take ln, then apply |
| 0⁰ | Take ln, then apply |
| ∞⁰ | Take ln, then apply |

### Examples

**Example 1:** lim(x→0) sin(x)/x

```
Direct: sin(0)/0 = 0/0 (indeterminate)

L'Hôpital: lim(x→0) cos(x)/1 = 1/1 = 1
```

**Example 2:** lim(x→∞) x/eˣ

```
Direct: ∞/∞ (indeterminate)

L'Hôpital: lim(x→∞) 1/eˣ = 0
```

**Example 3:** lim(x→0⁺) x·ln(x)

```
Direct: 0 · (-∞) (indeterminate)

Rewrite: lim(x→0⁺) ln(x)/(1/x) = ∞/∞

L'Hôpital: lim(x→0⁺) (1/x)/(-1/x²) = lim(x→0⁺) -x = 0
```

**Example 4:** lim(x→0⁺) xˣ

```
Direct: 0⁰ (indeterminate)

Let y = xˣ, then ln(y) = x·ln(x)

lim(x→0⁺) ln(y) = lim(x→0⁺) x·ln(x) = 0 (from Example 3)

So lim(x→0⁺) y = e⁰ = 1

Answer: lim(x→0⁺) xˣ = 1
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solve, lambdify, limit, oo, sin, cos, exp, ln

x = symbols('x')

# === Critical Points and Classification ===

def analyze_function(f_expr, x_range=(-5, 5)):
    """Complete function analysis"""
    f_prime = diff(f_expr, x)
    f_double = diff(f_prime, x)
    
    # Find critical points
    critical_points = solve(f_prime, x)
    
    print(f"f(x) = {f_expr}")
    print(f"f'(x) = {f_prime}")
    print(f"f''(x) = {f_double}")
    print(f"\nCritical points: {critical_points}")
    
    # Classify each critical point
    print("\nClassification:")
    for cp in critical_points:
        if cp.is_real:
            second_deriv_val = f_double.subs(x, cp)
            func_val = f_expr.subs(x, cp)
            
            if second_deriv_val > 0:
                classification = "Local MINIMUM"
            elif second_deriv_val < 0:
                classification = "Local MAXIMUM"
            else:
                classification = "Test inconclusive (check first derivative)"
            
            print(f"  x = {cp}: f(x) = {func_val}, {classification}")
    
    # Plot
    f_func = lambdify(x, f_expr, 'numpy')
    f_prime_func = lambdify(x, f_prime, 'numpy')
    
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(x_vals, f_func(x_vals), 'b-', linewidth=2, label='f(x)')
    ax1.set_title(f'Function: {f_expr}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax2.plot(x_vals, f_prime_func(x_vals), 'r-', linewidth=2, label="f'(x)")
    ax2.set_title(f"Derivative: {f_prime}")
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

# Example 1: Cubic function
analyze_function(x**3 - 3*x**2 - 9*x + 5)

# Example 2: Rational function
analyze_function(x**2 / (x**2 + 1))

# === Optimization Problem Solver ===

def optimize_rectangle_area(perimeter):
    """Find rectangle with maximum area for given perimeter"""
    # A(w) = w * (P/2 - w) = P*w/2 - w²
    w = symbols('w')
    P = perimeter
    
    A = P*w/2 - w**2
    A_prime = diff(A, w)
    A_double = diff(A_prime, w)
    
    # Critical point
    w_optimal = solve(A_prime, w)[0]
    h_optimal = P/2 - w_optimal
    
    print(f"Perimeter = {perimeter}")
    print(f"Optimal width: {w_optimal}")
    print(f"Optimal height: {h_optimal}")
    print(f"Maximum area: {A.subs(w, w_optimal)}")
    
    return float(w_optimal), float(h_optimal)

optimize_rectangle_area(20)

# === L'Hôpital's Rule Demonstrator ===

def demonstrate_lhopital(f_expr, g_expr, a):
    """Demonstrate L'Hôpital's rule"""
    f_prime = diff(f_expr, x)
    g_prime = diff(g_expr, x)
    
    # Check if indeterminate
    f_at_a = limit(f_expr, x, a)
    g_at_a = limit(g_expr, x, a)
    
    print(f"lim(x→{a}) {f_expr}/{g_expr}")
    print(f"Direct substitution: {f_at_a}/{g_at_a}")
    
    # Apply L'Hôpital
    if (f_at_a == 0 and g_at_a == 0) or (abs(f_at_a) == oo and abs(g_at_a) == oo):
        print("\nIndeterminate form! Applying L'Hôpital's Rule...")
        print(f"lim(x→{a}) {f_prime}/{g_prime}")
        
        result = limit(f_prime/g_prime, x, a)
        print(f"Result: {result}")
    else:
        print("\nNot indeterminate, no need for L'Hôpital")

# Example
demonstrate_lhopital(sin(x), x, 0)
demonstrate_lhopital(x, exp(x), oo)

# === Related Rates Calculator ===

def expanding_circle_problem(dr_dt, r):
    """How fast is area increasing?"""
    # A = πr²
    # dA/dt = 2πr · dr/dt
    
    dA_dt = 2 * np.pi * r * dr_dt
    
    print(f"Circle expanding at dr/dt = {dr_dt} cm/s")
    print(f"When r = {r} cm:")
    print(f"dA/dt = {dA_dt:.4f} cm²/s")
    
    return dA_dt

expanding_circle_problem(dr_dt=2, r=5)

# === Curve Sketching Helper ===

def curve_sketching_helper(f_expr):
    """Helper for complete curve sketching"""
    from sympy import limit, oo
    
    f_prime = diff(f_expr, x)
    f_double = diff(f_prime, x)
    
    print("=" * 60)
    print(f"CURVE SKETCHING: f(x) = {f_expr}")
    print("=" * 60)
    
    # 1. Domain
    print("\n1. DOMAIN: All real x (unless rational with zeros in denominator)")
    
    # 2. Intercepts
    y_intercept = f_expr.subs(x, 0)
    x_intercepts = solve(f_expr, x)
    print(f"\n2. INTERCEPTS:")
    print(f"   y-intercept: (0, {y_intercept})")
    print(f"   x-intercepts: {x_intercepts}")
    
    # 3. Critical points
    critical_points = solve(f_prime, x)
    print(f"\n3. CRITICAL POINTS: {critical_points}")
    
    # 4. Classification
    print("\n4. CLASSIFICATION:")
    for cp in critical_points:
        if cp.is_real:
            val = f_double.subs(x, cp)
            if val > 0:
                print(f"   x = {cp}: Local MIN")
            elif val < 0:
                print(f"   x = {cp}: Local MAX")
    
    # 5. Inflection points
    inflection_points = solve(f_double, x)
    print(f"\n5. INFLECTION POINTS: {inflection_points}")
    
    # 6. Asymptotes (for rational functions)
    print("\n6. ASYMPTOTES:")
    print(f"   lim(x→∞) f(x) = {limit(f_expr, x, oo)}")
    print(f"   lim(x→-∞) f(x) = {limit(f_expr, x, -oo)}")
    
    # Plot
    f_func = lambdify(x, f_expr, 'numpy')
    x_vals = np.linspace(-5, 5, 400)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, f_func(x_vals), 'b-', linewidth=2)
    plt.title(f'Curve Sketch: f(x) = {f_expr}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.show()

# Example
curve_sketching_helper(x**3 - 3*x)
```

---

## 📊 Summary Table

| Concept | Formula/Test | Use Case |
|---------|-------------|----------|
| **Critical Point** | f'(c) = 0 or undefined | Find extrema candidates |
| **First Derivative Test** | Sign change of f' | Classify critical points |
| **Second Derivative Test** | f''(c) > 0 → min, f''(c) < 0 → max | Quick classification |
| **Concavity** | f'' > 0 → up, f'' < 0 → down | Curve shape |
| **Inflection Point** | f'' = 0 and sign change | Where concavity changes |
| **MVT** | f'(c) = [f(b)-f(a)]/(b-a) | Theoretical foundation |
| **L'Hôpital's Rule** | lim f/g = lim f'/g' | Evaluate indeterminate forms |

---

## 🎯 ML Applications

| Application | Derivative Concept |
|-------------|-------------------|
| **Gradient Descent** | Finding critical points (∇L = 0) |
| **Loss Minimization** | Optimization using derivatives |
| **Learning Rate Tuning** | Related rates of change |
| **Convergence Analysis** | Limits and asymptotic behavior |
| **Second-Order Methods** | Using f'' (Hessian) for faster convergence |

---

## ❓ Quick Check Questions

1. What's the difference between critical and stationary points?
2. When does the second derivative test fail?
3. How do you solve related rates problems?
4. What are the conditions for L'Hôpital's rule?
5. State the Mean Value Theorem.
6. What's the strategy for optimization word problems?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Critical vs Stationary:**
   - Critical: f'(c) = 0 OR f'(c) undefined
   - Stationary: Only f'(c) = 0
   - All stationary are critical, not vice versa

2. **Second derivative test fails when:**
   - f''(c) = 0 (inconclusive)
   - Use first derivative test instead

3. **Related rates strategy:**
   - Draw diagram, write equation
   - Differentiate wrt time
   - Substitute known values
   - Solve for unknown rate

4. **L'Hôpital's conditions:**
   - 0/0 or ∞/∞ form
   - f and g differentiable
   - lim f'/g' must exist

5. **MVT:**
   - If f continuous on [a,b] and differentiable on (a,b)
   - Then ∃c ∈ (a,b) where f'(c) = [f(b)-f(a)]/(b-a)

6. **Optimization strategy:**
   - Define variables, write objective
   - Use constraints to reduce to one variable
   - Find critical points
   - Verify max/min
   - Answer original question

</details>
---

**Status:** ✅ Complete
**Next:** Partial Derivatives
