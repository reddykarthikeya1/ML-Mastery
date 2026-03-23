# Calculus - Practice Problems

## Topic 1: Functions and Limits

### Level 1: Basic

**1.1** Evaluate the following limits:
- a) lim(x→3) (2x + 1)
- b) lim(x→0) (x² - 4x + 3)
- c) lim(x→2) (x² - 4)/(x - 2)

**1.2** Find the vertical and horizontal asymptotes:
- a) f(x) = 1/(x - 3)
- b) f(x) = (2x + 1)/(x - 1)
- c) f(x) = (x² - 1)/(x² + 1)

**1.3** Determine if the function is continuous at the given point:
- a) f(x) = (x² - 1)/(x - 1) at x = 1
- b) f(x) = |x|/x at x = 0
- c) f(x) = 1/x² at x = 0

---

### Level 2: Intermediate

**1.4** Use the Squeeze Theorem to prove:
lim(x→0) x²·sin(1/x) = 0

**1.5** Find the value of k that makes the function continuous:
```
        { x² + k,    x < 2
f(x) = {
        { 3x - 1,    x ≥ 2
```

**1.6** Prove using the ε-δ definition:
lim(x→2) (3x - 1) = 5

**1.7** Python Practice:
```python
import numpy as np

# Numerically approximate the limit
def approximate_limit(f, a, n_points=10):
    h_values = [10**(-i) for i in range(1, n_points+1)]
    
    print(f"Approximating lim(x→{a}) f(x):")
    for h in h_values:
        left = f(a - h)
        right = f(a + h)
        print(f"h={h:.0e}: left={left:.10f}, right={right:.10f}")

# Test with lim(x→0) sin(x)/x
approximate_limit(lambda x: np.sin(x)/x if x != 0 else 1, 0)
```

---

## Topic 2: Differentiation

### Level 1: Basic

**2.1** Find the derivative using the power rule:
- a) f(x) = x⁵
- b) f(x) = 3x⁴ - 2x² + 7
- c) f(x) = √x

**2.2** Find the derivative using product rule:
- a) f(x) = x²·sin(x)
- b) f(x) = eˣ·ln(x)
- c) f(x) = (x³ + 1)(x² - 2x)

**2.3** Find the derivative using chain rule:
- a) f(x) = sin(x²)
- b) f(x) = e^(3x+1)
- c) f(x) = ln(cos(x))

---

### Level 2: Intermediate

**2.4** Find dy/dx using implicit differentiation:
- a) x² + y² = 25
- b) xy + sin(y) = x²
- c) x³ + y³ = 6xy (Folium of Descartes)

**2.5** Use logarithmic differentiation:
- a) y = xˣ (x > 0)
- b) y = (x+1)²·√(x-1)/(x+2)³
- c) y = (sin x)^(cos x)

**2.6** Find the equation of the tangent line:
- a) y = x² at point (2, 4)
- b) y = eˣ at point (0, 1)
- c) y = ln(x) at point (e, 1)

**2.7** Python Practice:
```python
import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2*h)

# Compare numerical and analytical derivatives
f = lambda x: x**3 - 2*x**2 + x - 1
f_prime_analytical = lambda x: 3*x**2 - 4*x + 1

x_vals = np.linspace(-2, 3, 100)
numerical_derivs = [numerical_derivative(f, x) for x in x_vals]
analytical_derivs = [f_prime_analytical(x) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, numerical_derivs, 'b--', label='Numerical')
plt.plot(x_vals, analytical_derivs, 'r-', label='Analytical')
plt.title('Numerical vs Analytical Derivative')
plt.legend()
plt.grid(True)
plt.show()
```

---

### Level 3: Advanced

**2.8** Prove Rolle's Theorem for f(x) = x² - 4x + 3 on [1, 3].

**2.9** Use the Mean Value Theorem to prove:
If f'(x) = 0 for all x, then f is constant.

**2.10** Find all critical points and classify them:
- a) f(x) = x³ - 3x² + 2
- b) f(x) = x·e^(-x)
- c) f(x) = sin(x) + cos(x) on [0, 2π]

---

## Topic 3: Gradient and Directional Derivatives

### Level 1: Basic

**3.1** Find the gradient of each function:
- a) f(x, y) = x² + 2xy + y²
- b) f(x, y, z) = x²y + yz²
- c) f(x, y) = eˣ·cos(y)

**3.2** Find the directional derivative at the given point in the given direction:
- a) f(x, y) = x² + y² at (1, 2) in direction (3, 4)
- b) f(x, y) = xy at (2, 3) in direction (1, -1)

**3.3** Find the direction of maximum increase:
- a) f(x, y) = x² - y² at (1, 1)
- b) f(x, y, z) = x + 2y + 3z at (0, 0, 0)

---

### Level 2: Intermediate

**3.4** Prove that the gradient is perpendicular to level curves.

**3.5** Find and classify all critical points:
- a) f(x, y) = x² + y²
- b) f(x, y) = x² - y²
- c) f(x, y) = x³ - 3xy + y³

**3.6** Use Lagrange multipliers to find extrema:
- a) f(x, y) = x² + y² subject to x + y = 1
- b) f(x, y, z) = xyz subject to x² + y² + z² = 1

**3.7** Python Practice - Gradient Descent:
```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, start, lr=0.1, n_iter=50):
    """Implement gradient descent"""
    path = [np.array(start, dtype=float)]
    
    for _ in range(n_iter):
        current = path[-1]
        grad = grad_f(current)
        next_point = current - lr * grad
        path.append(next_point)
    
    return np.array(path)

# Test on f(x, y) = x² + y²
f = lambda x: x[0]**2 + x[1]**2
grad_f = lambda x: np.array([2*x[0], 2*x[1]])

path = gradient_descent(f, grad_f, start=[-3, -2], lr=0.1, n_iter=20)

# Plot the path
plt.figure(figsize=(8, 8))
plt.plot(path[:, 0], path[:, 1], 'b-o', linewidth=2, markersize=5)
plt.plot(path[0, 0], path[0, 1], 'go', markersize=15, label='Start')
plt.plot(path[-1, 0], path[-1, 1], 'r*', markersize=20, label='End')
plt.title('Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

print(f"Final position: {path[-1]}")
print(f"Final value: {f(path[-1]):.10f}")
```

---

### Level 3: Advanced

**3.8** Derive the multivariate Taylor expansion up to second order.

**3.9** Prove that for a convex function, any local minimum is a global minimum.

**3.10** Implement Newton's method and compare with gradient descent:
```python
def newtons_method(f, grad_f, hess_f, start, n_iter=10):
    """Implement Newton's method for optimization"""
    x = np.array(start, dtype=float)
    path = [x.copy()]
    
    for _ in range(n_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Newton step: x - H^{-1} * grad
        try:
            step = np.linalg.solve(hess, grad)
            x = x - step
            path.append(x.copy())
        except np.linalg.LinAlgError:
            print("Hessian is singular!")
            break
    
    return np.array(path)

# Test on Rosenbrock function
f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
grad_f = lambda x: np.array([
    -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
    200*(x[1] - x[0]**2)
])
hess_f = lambda x: np.array([
    [2 - 400*x[1] + 1200*x[0]**2, -400*x[0]],
    [-400*x[0], 200]
])

# Compare methods
start = [-1, 1]
newton_path = newtons_method(f, grad_f, hess_f, start, n_iter=10)
print(f"Newton's method final: {newton_path[-1]}")
print(f"Newton's method value: {f(newton_path[-1]):.10f}")
```

---

## Topic 4: Integration

### Level 1: Basic

**4.1** Evaluate the indefinite integrals:
- a) ∫ x³ dx
- b) ∫ (2x + 1)² dx
- c) ∫ e^(2x) dx

**4.2** Evaluate using substitution:
- a) ∫ 2x·e^(x²) dx
- b) ∫ sin(2x)·cos(2x) dx
- c) ∫ x/√(1-x²) dx

**4.3** Evaluate the definite integrals:
- a) ∫₀¹ x² dx
- b) ∫₀^π sin(x) dx
- c) ∫₁^e 1/x dx

---

### Level 2: Intermediate

**4.4** Evaluate using integration by parts:
- a) ∫ x·eˣ dx
- b) ∫ x²·ln(x) dx
- c) ∫ eˣ·sin(x) dx

**4.5** Evaluate using partial fractions:
- a) ∫ 1/(x² - 1) dx
- b) ∫ (x+1)/(x² + 3x + 2) dx
- c) ∫ 1/(x³ - x) dx

**4.6** Find the area between curves:
- a) y = x² and y = x
- b) y = sin(x) and y = cos(x) on [0, π/4]
- c) y = eˣ and y = x + 1 on [0, 1]

---

## Topic 5: Advanced Applications

### Level 3: Challenge Problems

**5.1** Prove the Fundamental Theorem of Calculus.

**5.2** Derive the formula for arc length of a curve.

**5.3** Use Taylor series to approximate:
- a) ∫₀¹ e^(-x²) dx (error function)
- b) lim(x→0) (sin(x) - x)/x³

**5.4** Python Challenge - Numerical Integration:
```python
import numpy as np

def trapezoidal_rule(f, a, b, n=100):
    """Numerical integration using trapezoidal rule"""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (np.sum(y) - 0.5*(y[0] + y[-1]))

def simpsons_rule(f, a, b, n=100):
    """Numerical integration using Simpson's rule"""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))

# Test on ∫₀¹ x² dx = 1/3
f = lambda x: x**2

trap_result = trapezoidal_rule(f, 0, 1, n=100)
simp_result = simpsons_rule(f, 0, 1, n=100)
exact = 1/3

print(f"Trapezoidal: {trap_result:.10f}, Error: {abs(trap_result - exact):.2e}")
print(f"Simpson's: {simp_result:.10f}, Error: {abs(simp_result - exact):.2e}")
```

---

## Solutions (Attempt First!)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
```
a) 7
b) 3
c) lim(x→2) (x-2)(x+2)/(x-2) = lim(x→2) (x+2) = 4
```

### 1.2
```
a) Vertical: x=3, Horizontal: y=0
b) Vertical: x=1, Horizontal: y=2
c) No vertical, Horizontal: y=1
```

### 2.1
```
a) 5x⁴
b) 12x³ - 4x
c) 1/(2√x)
```

### 2.2
```
a) 2x·sin(x) + x²·cos(x)
b) eˣ·ln(x) + eˣ/x
c) (3x² + 2x)(x² - 2x) + (x³ + 1)(2x - 2)
```

### 2.3
```
a) 2x·cos(x²)
b) 3e^(3x+1)
c) -tan(x)
```

### 3.1
```
a) ∇f = (2x + 2y, 2x + 2y)
b) ∇f = (2xy, x² + z², 2yz)
c) ∇f = (eˣ·cos(y), -eˣ·sin(y))
```

### 3.2
```
a) D_u f = (2, 4) · (3/5, 4/5) = 22/5 = 4.4
b) D_u f = (3, 2) · (1/√2, -1/√2) = 1/√2
```

### 4.1
```
a) x⁴/4 + C
b) (2x+1)³/6 + C
c) e^(2x)/2 + C
```

### 4.2
```
a) e^(x²) + C
b) -cos²(2x)/4 + C
c) -√(1-x²) + C
```

</details>

---

## 📝 Notes Section

Use this space for additional problems you encounter:

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
