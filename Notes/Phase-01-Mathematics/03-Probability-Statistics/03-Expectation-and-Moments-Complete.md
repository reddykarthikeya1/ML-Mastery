# 1.3.3 Expectation and Moments

## 🎯 Quick Overview
- **Expected value**: Long-run average, center of distribution
- **Variance**: Spread around the mean
- **Moments**: Generalized measures of distribution shape
- **Foundation for**: Statistical inference, risk assessment, ML loss functions

---

## 1. Expected Value (Mean)

### Definition for Discrete RV

```
E[X] = Σ x · p(x)

Sum over all possible values of X
```

### Definition for Continuous RV

```
E[X] = ∫ x · f(x) dx

Over the support (where f(x) > 0)
```

### Alternative Notation

```
μ = E[X] = μ_X
```

### Interpretation

| Perspective | Meaning |
|-------------|---------|
| **Physical** | Center of mass/balance point |
| **Long-run** | Average over many trials |
| **Weighted** | Probability-weighted average |

### Examples

**Example 1: Fair Die**
```
X = outcome of fair die roll

E[X] = 1·(1/6) + 2·(1/6) + 3·(1/6) + 4·(1/6) + 5·(1/6) + 6·(1/6)
     = (1+2+3+4+5+6)/6
     = 21/6
     = 3.5
```

**Example 2: Bernoulli(p)**
```
X = { 1 with probability p
    { 0 with probability 1-p

E[X] = 1·p + 0·(1-p) = p

The parameter p IS the expected value!
```

**Example 3: Uniform(a, b)**
```
X ~ Uniform(a, b)

E[X] = ∫ₐᵇ x · (1/(b-a)) dx
     = (1/(b-a)) · [x²/2]ₐᵇ
     = (b² - a²) / (2(b-a))
     = (a + b) / 2

The midpoint of the interval!
```

---

## 2. Linearity of Expectation

### Properties

**Constant:**
```
E[c] = c
```

**Scaling and Shift:**
```
E[aX + b] = a·E[X] + b
```

**Sum of RVs:**
```
E[X + Y] = E[X] + E[Y]

TRUE FOR ANY X, Y (independent or not!)
```

**General Linear Combination:**
```
E[a₁X₁ + a₂X₂ + ... + aₙXₙ] = a₁E[X₁] + a₂E[X₂] + ... + aₙE[Xₙ]
```

### Examples

**Example 4: Sum of Dice**
```
Roll 2 dice, X = sum of outcomes

X = X₁ + X₂ where X₁, X₂ are individual rolls

E[X₁] = E[X₂] = 3.5

E[X] = E[X₁ + X₂] = E[X₁] + E[X₂] = 3.5 + 3.5 = 7

No need to compute full distribution!
```

**Example 5: Binomial Expectation**
```
X ~ Binomial(n, p)

X = X₁ + X₂ + ... + Xₙ where Xᵢ ~ Bernoulli(p)

E[Xᵢ] = p for each i

E[X] = n · p

Much easier than summing over all outcomes!
```

---

## 3. Expected Value of Functions

### LOTUS (Law of the Unconscious Statistician)

**For Discrete RV:**
```
E[g(X)] = Σ g(x) · p(x)
```

**For Continuous RV:**
```
E[g(X)] = ∫ g(x) · f(x) dx
```

### Important: NOT E[g(X)] = g(E[X])!

```
E[X²] ≠ (E[X])²  in general

Example: Fair die
E[X] = 3.5
(E[X])² = 12.25

E[X²] = (1+4+9+16+25+36)/6 = 91/6 = 15.17

E[X²] ≠ (E[X])²
```

### Examples

**Example 6: E[X²] for Bernoulli**
```
X ~ Bernoulli(p)

X² = X (since X is 0 or 1)

E[X²] = E[X] = p
```

**Example 7: E[e^X] for Exponential**
```
X ~ Exponential(λ)

E[e^X] = ∫₀^∞ e^x · λe^(-λx) dx
       = λ ∫₀^∞ e^((1-λ)x) dx

For λ > 1:
       = λ / (λ - 1)
```

---

## 4. Variance and Standard Deviation

### Definition

**Variance:**
```
Var(X) = E[(X - μ)²]

where μ = E[X]
```

**Standard Deviation:**
```
SD(X) = σ = √Var(X)
```

### Computational Formula

```
Var(X) = E[X²] - (E[X])²

Proof:
Var(X) = E[(X - μ)²]
       = E[X² - 2μX + μ²]
       = E[X²] - 2μE[X] + μ²
       = E[X²] - 2μ·μ + μ²
       = E[X²] - μ²
       = E[X²] - (E[X])²
```

### Properties

| Property | Formula |
|----------|---------|
| **Non-negative** | Var(X) ≥ 0 |
| **Constant** | Var(c) = 0 |
| **Scaling** | Var(aX + b) = a²Var(X) |
| **Shift invariant** | Var(X + b) = Var(X) |

### Examples

**Example 8: Variance of Fair Die**
```
E[X] = 3.5
E[X²] = 91/6 = 15.167

Var(X) = E[X²] - (E[X])²
       = 15.167 - 12.25
       = 2.917

SD(X) = √2.917 ≈ 1.708
```

**Example 9: Variance of Bernoulli**
```
X ~ Bernoulli(p)

E[X] = p
E[X²] = p (since X² = X)

Var(X) = p - p² = p(1-p)

Maximum variance at p = 0.5:
Var(X) = 0.5 × 0.5 = 0.25
```

---

## 5. Properties of Variance

### Sum of Independent RVs

**If X and Y are independent:**
```
Var(X + Y) = Var(X) + Var(Y)
```

**General (with covariance):**
```
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)

Var(X - Y) = Var(X) + Var(Y) - 2Cov(X, Y)
```

### Examples

**Example 10: Sum of n iid Variables**
```
X₁, X₂, ..., Xₙ are iid with variance σ²

S = X₁ + X₂ + ... + Xₙ

Var(S) = n · σ²

(Since independent, variances add)
```

**Example 11: Sample Mean Variance**
```
X̄ = (X₁ + X₂ + ... + Xₙ) / n

Var(X̄) = Var((1/n) · ΣXᵢ)
       = (1/n²) · Var(ΣXᵢ)
       = (1/n²) · n · σ²
       = σ² / n

Larger sample size → smaller variance!
```

---

## 6. Covariance

### Definition

```
Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]

Alternative formula:
Cov(X, Y) = E[XY] - E[X]E[Y]
```

### Interpretation

| Cov(X, Y) | Relationship |
|-----------|--------------|
| > 0 | Positive linear relationship |
| < 0 | Negative linear relationship |
| = 0 | No LINEAR relationship |

### Properties

| Property | Formula |
|----------|---------|
| **Symmetry** | Cov(X, Y) = Cov(Y, X) |
| **Self-covariance** | Cov(X, X) = Var(X) |
| **Scaling** | Cov(aX, bY) = ab·Cov(X, Y) |
| **Bilinearity** | Cov(aX+b, cY+d) = ac·Cov(X, Y) |

### Examples

**Example 12: Covariance Calculation**
```
Joint distribution of X and Y:

     Y=0   Y=1
X=0  0.1   0.2
X=1  0.3   0.4

Marginals:
P(X=0) = 0.3, P(X=1) = 0.7
P(Y=0) = 0.4, P(Y=1) = 0.6

E[X] = 0·0.3 + 1·0.7 = 0.7
E[Y] = 0·0.4 + 1·0.6 = 0.6

E[XY] = 0·0·0.1 + 0·1·0.2 + 1·0·0.3 + 1·1·0.4 = 0.4

Cov(X, Y) = E[XY] - E[X]E[Y]
          = 0.4 - 0.7·0.6
          = 0.4 - 0.42
          = -0.02

Slight negative relationship
```

---

## 7. Correlation Coefficient

### Definition

```
ρ(X, Y) = Corr(X, Y) = Cov(X, Y) / (σₓ · σᵧ)

where σₓ = SD(X), σᵧ = SD(Y)
```

### Properties

| Property | Value |
|----------|-------|
| **Bounds** | -1 ≤ ρ ≤ 1 |
| **Perfect positive** | ρ = 1 |
| **Perfect negative** | ρ = -1 |
| **No linear relationship** | ρ = 0 |
| **Scale invariant** | Corr(aX+b, cY+d) = ±Corr(X,Y) |

### Interpretation

| ρ Value | Strength |
|---------|----------|
| |ρ| ≥ 0.8 | Strong |
| 0.5 ≤ |ρ| < 0.8 | Moderate |
| 0.3 ≤ |ρ| < 0.5 | Weak |
| |ρ| < 0.3 | Very weak/none |

### Key Insight

```
Correlation ≠ Causation!

High correlation could mean:
1. X causes Y
2. Y causes X
3. Common cause Z affects both
4. Coincidence (especially with small samples)
```

---

## 8. Moments

### Raw Moments

**k-th raw moment:**
```
μ'ₖ = E[Xᵏ]
```

### Central Moments

**k-th central moment:**
```
μₖ = E[(X - μ)ᵏ]
```

### Summary of Common Moments

| k | Raw Moment | Central Moment | Name |
|---|------------|----------------|------|
| 1 | E[X] | 0 | Mean |
| 2 | E[X²] | Var(X) | Variance |
| 3 | E[X³] | E[(X-μ)³] | Related to skewness |
| 4 | E[X⁴] | E[(X-μ)⁴] | Related to kurtosis |

---

## 9. Skewness

### Definition

```
Skewness = γ₁ = E[(X - μ)³] / σ³

Standardized third central moment
```

### Interpretation

| Skewness | Shape | Tail |
|----------|-------|------|
| > 0 | Right-skewed | Longer right tail |
| < 0 | Left-skewed | Longer left tail |
| = 0 | Symmetric | Equal tails |

### Examples

| Distribution | Skewness |
|--------------|----------|
| Normal | 0 |
| Exponential(λ) | 2 |
| Chi-squared(k) | √(8/k) |
| Binomial(n,p) | (1-2p)/√(np(1-p)) |

---

## 10. Kurtosis

### Definition

```
Kurtosis = β₂ = E[(X - μ)⁴] / σ⁴

Standardized fourth central moment
```

**Excess Kurtosis:**
```
Excess Kurtosis = Kurtosis - 3
```

### Interpretation

| Kurtosis | Excess | Shape | Tails |
|----------|--------|-------|-------|
| > 3 | > 0 | Leptokurtic | Heavy |
| = 3 | = 0 | Mesokurtic | Normal |
| < 3 | < 0 | Platykurtic | Light |

### Examples

| Distribution | Kurtosis | Excess |
|--------------|----------|--------|
| Normal | 3 | 0 |
| Uniform | 1.8 | -1.2 |
| Exponential | 9 | 6 |
| Laplace | 6 | 3 |

---

## 11. Moment Generating Functions (MGF)

### Definition

```
M_X(t) = E[e^(tX)]

Discrete: M_X(t) = Σ e^(tx) · p(x)
Continuous: M_X(t) = ∫ e^(tx) · f(x) dx
```

### Key Property

**Moments from MGF:**
```
M'_X(0) = E[X]
M''_X(0) = E[X²]
M⁽ᵏ⁾_X(0) = E[Xᵏ]

Differentiate k times, evaluate at t=0
```

### Examples

**Bernoulli(p):**
```
M_X(t) = (1-p)·e^(t·0) + p·e^(t·1)
       = (1-p) + p·e^t

M'_X(t) = p·e^t
M'_X(0) = p = E[X] ✓
```

**Exponential(λ):**
```
M_X(t) = λ / (λ - t)  for t < λ

M'_X(t) = λ / (λ - t)²
M'_X(0) = λ/λ² = 1/λ = E[X] ✓
```

### Uniqueness Property

```
If M_X(t) = M_Y(t) for all t in a neighborhood of 0,
then X and Y have the same distribution.

Useful for proving distributions are equal!
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Expected Value and Variance ===

print("=" * 60)
print("EXPECTED VALUE AND VARIANCE")
print("=" * 60)

# Discrete: Fair die
x_die = np.array([1, 2, 3, 4, 5, 6])
p_die = np.array([1/6] * 6)

E_die = np.sum(x_die * p_die)
E_die2 = np.sum(x_die**2 * p_die)
Var_die = E_die2 - E_die**2

print(f"\nFair Die:")
print(f"E[X] = {E_die:.4f}")
print(f"E[X²] = {E_die2:.4f}")
print(f"Var(X) = {Var_die:.4f}")
print(f"SD(X) = {np.sqrt(Var_die):.4f}")

# Continuous: Normal distribution
mu, sigma = 100, 15
normal = stats.norm(mu, sigma)

print(f"\nNormal(μ={mu}, σ={sigma}):")
print(f"E[X] = {normal.mean():.4f}")
print(f"Var(X) = {normal.var():.4f}")
print(f"SD(X) = {normal.std():.4f}")

# === Covariance and Correlation ===

print("\n" + "=" * 60)
print("COVARIANCE AND CORRELATION")
print("=" * 60)

# Generate correlated data
np.random.seed(42)
n = 1000

# Method 1: Multivariate normal
mean = [0, 0]
cov_matrix = [[1, 0.7], [0.7, 1]]  # Correlation = 0.7
data = np.random.multivariate_normal(mean, cov_matrix, n)

X = data[:, 0]
Y = data[:, 1]

# Calculate covariance and correlation
cov_XY = np.cov(X, Y)[0, 1]
corr_XY = np.corrcoef(X, Y)[0, 1]

print(f"Simulated data (n={n}, ρ=0.7):")
print(f"Sample Cov(X,Y) = {cov_XY:.4f}")
print(f"Sample Corr(X,Y) = {corr_XY:.4f}")

# Verify relationship
print(f"\nVerification:")
print(f"Cov(X,Y) / (SD(X)·SD(Y)) = {cov_XY / (np.std(X) * np.std(Y)):.4f}")
print(f"Should equal correlation: {corr_XY:.4f}")

# === Moment Calculations ===

print("\n" + "=" * 60)
print("MOMENTS AND SKEWNESS/KURTOSIS")
print("=" * 60)

# Compare different distributions
distributions = {
    'Normal(0,1)': stats.norm(0, 1),
    'Exponential(1)': stats.expon(0, 1),
    'Uniform(0,1)': stats.uniform(0, 1),
    'Chi-squared(5)': stats.chi2(5)
}

print(f"{'Distribution':<20} {'Mean':<10} {'Var':<10} {'Skew':<10} {'Kurt':<10}")
print("-" * 60)

for name, dist in distributions.items():
    samples = dist.rvs(100000)
    
    mean = np.mean(samples)
    var = np.var(samples)
    skew = stats.skew(samples)
    kurt = stats.kurtosis(samples)  # This gives excess kurtosis
    
    print(f"{name:<20} {mean:<10.4f} {var:<10.4f} {skew:<10.4f} {kurt:<10.4f}")

# === Visualization of Moments ===

def visualize_moments():
    """Visualize how moments describe distribution shape"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Different distributions to show moments
    dists = [
        (stats.norm(0, 1), 'Normal (symmetric)', 'blue'),
        (stats.skewnorm(5, 0, 1), 'Skewed right', 'red'),
        (stats.laplace(0, 1), 'Heavy tails', 'green'),
    ]
    
    x = np.linspace(-5, 5, 400)
    
    for i, (dist, name, color) in enumerate(dists):
        ax = axes[i // 2, i % 2]
        
        pdf = dist.pdf(x)
        mean = dist.mean()
        std = dist.std()
        skew = stats.skew(dist.rvs(10000))
        kurt = stats.kurtosis(dist.rvs(10000))
        
        ax.plot(x, pdf, color=color, linewidth=2, label=name)
        ax.axvline(mean, color='black', linestyle='--', label=f'Mean = {mean:.2f}')
        ax.fill_between(x, pdf, alpha=0.3, color=color)
        
        ax.set_title(f'{name}\nSkew={skew:.2f}, Ex.Kurt={kurt:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Fourth subplot: Compare variances
    ax = axes[1, 1]
    x = np.linspace(-5, 5, 400)
    
    for sigma, style in [(0.5, '-'), (1, '--'), (2, '-.')]:
        dist = stats.norm(0, sigma)
        pdf = dist.pdf(x)
        ax.plot(x, pdf, style, linewidth=2, label=f'σ = {sigma}')
    
    ax.set_title('Effect of Variance (spread)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_moments()

# === Linearity of Expectation Demo ===

def linearity_demo():
    """Demonstrate linearity of expectation"""
    
    print("\n" + "=" * 60)
    print("LINEARITY OF EXPECTATION DEMO")
    print("=" * 60)
    
    # Sum of random variables
    n_dice = [1, 2, 5, 10, 20]
    
    print(f"{'n dice':<10} {'E[Sum]':<15} {'n × 3.5':<15} {'Match'}")
    print("-" * 45)
    
    for n in n_dice:
        # Simulate
        rolls = np.random.randint(1, 7, size=(10000, n))
        sums = np.sum(rolls, axis=1)
        E_sum = np.mean(sums)
        
        theoretical = n * 3.5
        match = abs(E_sum - theoretical) < 0.1
        
        print(f"{n:<10} {E_sum:<15.4f} {theoretical:<15.4f} {'✓' if match else '✗'}")

linearity_demo()

# === Covariance Matrix Visualization ===

def visualize_covariance():
    """Visualize different covariance structures"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Different correlation values
    rhos = [0.9, 0.5, 0, -0.7]
    titles = ['Strong positive (ρ=0.9)', 
              'Moderate positive (ρ=0.5)',
              'No correlation (ρ=0)',
              'Negative (ρ=-0.7)']
    
    for ax, rho, title in zip(axes, rhos, titles):
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        data = np.random.multivariate_normal(mean, cov, 500)
        
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add correlation text
        actual_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        ax.text(0.05, 0.95, f'Sample ρ = {actual_corr:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

visualize_covariance()

# === Moment Generating Function ===

def mgf_examples():
    """Demonstrate MGF for different distributions"""
    
    print("\n" + "=" * 60)
    print("MOMENT GENERATING FUNCTIONS")
    print("=" * 60)
    
    # For exponential distribution
    lambda_param = 2
    
    # Theoretical MGF: M(t) = λ / (λ - t) for t < λ
    def mgf_exponential(t, lam):
        if t >= lam:
            return np.inf
        return lam / (lam - t)
    
    # Numerical MGF from samples
    samples = np.random.exponential(1/lambda_param, 100000)
    
    t_values = np.linspace(-1, 1.5, 50)
    
    theoretical_mgf = [mgf_exponential(t, lambda_param) for t in t_values]
    
    # Numerical estimate: E[e^(tX)]
    numerical_mgf = [np.mean(np.exp(t * samples)) for t in t_values]
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, theoretical_mgf, 'b-', linewidth=2, label='Theoretical MGF')
    plt.plot(t_values, numerical_mgf, 'r--', linewidth=2, label='Numerical estimate')
    plt.axvline(x=lambda_param, color='green', linestyle=':', label=f't = λ = {lambda_param}')
    plt.title(f'MGF of Exponential(λ={lambda_param})')
    plt.xlabel('t')
    plt.ylabel('M(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 10)
    plt.show()
    
    # Verify moments from MGF
    print(f"\nExponential(λ={lambda_param}):")
    print(f"Theoretical E[X] = {1/lambda_param:.4f}")
    print(f"Sample E[X] = {np.mean(samples):.4f}")
    print(f"\nTheoretical Var(X) = {1/lambda_param**2:.4f}")
    print(f"Sample Var(X) = {np.var(samples):.4f}")

mgf_examples()
```

---

## 📊 Summary Table

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| **Expected Value** | E[X] = Σ x·p(x) or ∫ x·f(x)dx | Center, mean |
| **Linearity** | E[aX + bY] = aE[X] + bE[Y] | Always true |
| **Variance** | Var(X) = E[(X-μ)²] = E[X²] - (E[X])² | Spread |
| **Covariance** | Cov(X,Y) = E[XY] - E[X]E[Y] | Linear relationship |
| **Correlation** | ρ = Cov(X,Y)/(σₓσᵧ) | Standardized covariance |
| **Skewness** | E[(X-μ)³]/σ³ | Asymmetry |
| **Kurtosis** | E[(X-μ)⁴]/σ⁴ | Tail heaviness |
| **MGF** | M(t) = E[e^(tX)] | Generates moments |

---

## 🎯 ML Applications

| Application | Moments Concept |
|-------------|-----------------|
| **Loss Functions** | MSE = E[(Y-Ŷ)²] |
| **Feature Scaling** | Standardization uses mean, variance |
| **PCA** | Uses covariance matrix |
| **Batch Normalization** | Normalizes using batch moments |
| **Uncertainty** | Variance of predictions |
| **Distribution Matching** | Match moments between distributions |

---

## ❓ Quick Check Questions

1. Why doesn't E[g(X)] = g(E[X]) in general?
2. State the linearity of expectation.
3. What's the computational formula for variance?
4. How are covariance and correlation related?
5. What does skewness measure?
6. How do you get moments from MGF?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **E[g(X)] ≠ g(E[X]):**
   - Expectation is linear, not nonlinear
   - Jensen's inequality: E[g(X)] ≥ g(E[X]) for convex g

2. **Linearity of expectation:**
   - E[aX + bY] = aE[X] + bE[Y]
   - True even if X, Y are dependent!

3. **Computational variance:**
   - Var(X) = E[X²] - (E[X])²
   - Often easier than definition

4. **Covariance vs Correlation:**
   - ρ = Cov(X,Y)/(σₓσᵧ)
   - Correlation is standardized (-1 to 1)

5. **Skewness:**
   - Measures asymmetry
   - Positive = right tail, Negative = left tail

6. **Moments from MGF:**
   - M⁽ᵏ⁾(0) = E[Xᵏ]
   - Differentiate k times, evaluate at t=0

</details>
---

**Status:** ✅ Complete
**Next:** Discrete Distributions
