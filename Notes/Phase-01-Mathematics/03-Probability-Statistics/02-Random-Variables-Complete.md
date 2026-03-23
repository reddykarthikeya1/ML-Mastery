# 1.3.2 Random Variables

## 🎯 Quick Overview
- **Random Variable**: Numerical outcome of random experiment
- **PMF/PDF**: Probability distribution
- **Expected value**: Long-run average
- **Foundation for**: All statistical inference, ML model outputs

---

## 1. Random Variables - Definition

### Formal Definition

A **random variable (RV)** is a function that assigns a real number to each outcome in the sample space.

```
X: S → ℝ

For each outcome s ∈ S, X(s) is a real number
```

### Notation Convention

| Notation | Meaning |
|----------|---------|
| **X, Y, Z** | Random variables (uppercase) |
| **x, y, z** | Specific values (lowercase) |
| **P(X = x)** | Probability that X takes value x |

### Example

```
Experiment: Flip 2 coins
Sample space: S = {HH, HT, TH, TT}

Random variable X = "number of heads":
X(HH) = 2
X(HT) = 1
X(TH) = 1
X(TT) = 0

Possible values: {0, 1, 2}
```

---

## 2. Discrete vs Continuous Random Variables

### Discrete Random Variables

**Definition:** Takes countable number of distinct values

**Examples:**
- Number of heads in n flips: {0, 1, 2, ..., n}
- Number of customers: {0, 1, 2, ...}
- Die roll: {1, 2, 3, 4, 5, 6}

### Continuous Random Variables

**Definition:** Takes uncountably many values (interval of real numbers)

**Examples:**
- Height of person: [0, ∞) or specific range
- Time until event: [0, ∞)
- Temperature: (-∞, ∞) or specific range

### Comparison

| Aspect | Discrete | Continuous |
|--------|----------|------------|
| **Values** | Countable | Uncountable (interval) |
| **Probability function** | PMF: P(X = x) | PDF: f(x) |
| **P(X = x)** | Can be > 0 | Always = 0 |
| **Probability of range** | Σ P(X = x) | ∫ f(x) dx |
| **CDF** | Step function | Continuous function |

---

## 3. Probability Mass Function (PMF)

### Definition

For discrete RV X, the **PMF** is:
```
p(x) = P(X = x)
```

### Properties

1. **Non-negativity:** p(x) ≥ 0 for all x
2. **Normalization:** Σ p(x) = 1 (sum over all possible x)
3. **Probability of event:** P(X ∈ A) = Σ_{x∈A} p(x)

### Example: Fair Die

```
X = outcome of fair die roll

PMF:
p(1) = 1/6
p(2) = 1/6
p(3) = 1/6
p(4) = 1/6
p(5) = 1/6
p(6) = 1/6

Check: Σ p(x) = 6 × (1/6) = 1 ✓

P(X ≥ 4) = p(4) + p(5) + p(6) = 3/6 = 0.5
```

### Example: Biased Coin

```
X = number of heads in 2 flips of biased coin (p = 0.6)

Possible values: {0, 1, 2}

p(0) = P(TT) = 0.4 × 0.4 = 0.16
p(1) = P(HT or TH) = 0.6×0.4 + 0.4×0.6 = 0.48
p(2) = P(HH) = 0.6 × 0.6 = 0.36

Check: 0.16 + 0.48 + 0.36 = 1 ✓
```

---

## 4. Probability Density Function (PDF)

### Definition

For continuous RV X, the **PDF** f(x) satisfies:
```
P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx
```

### Properties

1. **Non-negativity:** f(x) ≥ 0 for all x
2. **Normalization:** ∫_{-∞}^{∞} f(x) dx = 1
3. **Probability of range:** P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx

### Important Notes

```
P(X = x) = 0 for any specific value x

Why? ∫_x^x f(x) dx = 0 (zero-width interval)

Therefore:
P(a ≤ X ≤ b) = P(a < X ≤ b) = P(a ≤ X < b) = P(a < X < b)
```

### Example: Uniform Distribution

```
X ~ Uniform(0, 1)

PDF:
f(x) = { 1,  0 ≤ x ≤ 1
       { 0,  otherwise

Check normalization:
∫₀¹ 1 dx = [x]₀¹ = 1 ✓

P(0.3 ≤ X ≤ 0.7) = ∫_{0.3}^{0.7} 1 dx = 0.4
```

### Example: Exponential Distribution

```
X ~ Exponential(λ = 2)

PDF:
f(x) = { 2e^(-2x),  x ≥ 0
       { 0,         x < 0

Check normalization:
∫₀^∞ 2e^(-2x) dx = [-e^(-2x)]₀^∞ = 0 - (-1) = 1 ✓

P(X > 1) = ∫₁^∞ 2e^(-2x) dx = [e^(-2x)]₁^∞ = e^(-2) ≈ 0.135
```

---

## 5. Cumulative Distribution Function (CDF)

### Definition

The **CDF** of random variable X is:
```
F(x) = P(X ≤ x)
```

### For Discrete RV

```
F(x) = Σ_{t ≤ x} p(t)

Step function, right-continuous
```

### For Continuous RV

```
F(x) = ∫_{-∞}^{x} f(t) dt

Continuous function
F'(x) = f(x) (derivative gives PDF)
```

### Properties (Both Types)

1. **Non-decreasing:** x₁ < x₂ → F(x₁) ≤ F(x₂)
2. **Limits:** F(-∞) = 0, F(∞) = 1
3. **Range probability:** P(a < X ≤ b) = F(b) - F(a)

### Example: Discrete CDF (Die Roll)

```
X = fair die roll

PMF: p(x) = 1/6 for x ∈ {1,2,3,4,5,6}

CDF:
F(1) = P(X ≤ 1) = 1/6
F(2) = P(X ≤ 2) = 2/6 = 1/3
F(3) = P(X ≤ 3) = 3/6 = 1/2
F(4) = P(X ≤ 4) = 4/6 = 2/3
F(5) = P(X ≤ 5) = 5/6
F(6) = P(X ≤ 6) = 1

P(2 < X ≤ 4) = F(4) - F(2) = 2/3 - 1/3 = 1/3
```

### Example: Continuous CDF (Uniform)

```
X ~ Uniform(0, 1)

PDF: f(x) = 1 for 0 ≤ x ≤ 1

CDF:
F(x) = ∫₀^x 1 dt = x  for 0 ≤ x ≤ 1

F(0) = 0
F(0.5) = 0.5
F(1) = 1

P(0.2 < X < 0.8) = F(0.8) - F(0.2) = 0.8 - 0.2 = 0.6
```

---

## 6. Expected Value (Mean)

### Definition for Discrete RV

```
E[X] = Σ x · p(x)

"Weighted average of all possible values"
```

### Definition for Continuous RV

```
E[X] = ∫ x · f(x) dx

Over the support of X
```

### Alternative Notation

```
μ = E[X]
```

### Properties

| Property | Formula |
|----------|---------|
| **Constant** | E[c] = c |
| **Linearity** | E[aX + b] = aE[X] + b |
| **Sum** | E[X + Y] = E[X] + E[Y] |
| **Function of X** | E[g(X)] = Σ g(x)p(x) or ∫ g(x)f(x)dx |

### Example: Die Roll

```
X = fair die roll

E[X] = Σ x · p(x)
     = 1·(1/6) + 2·(1/6) + 3·(1/6) + 4·(1/6) + 5·(1/6) + 6·(1/6)
     = (1+2+3+4+5+6)/6
     = 21/6
     = 3.5

Note: E[X] doesn't have to be a possible value!
```

### Example: Coin Flip (Bernoulli)

```
X = number of heads in 1 flip of coin with P(H) = p

PMF:
p(0) = 1-p
p(1) = p

E[X] = 0·(1-p) + 1·p = p

The parameter p IS the expected value!
```

### Example: Continuous Uniform

```
X ~ Uniform(a, b)

PDF: f(x) = 1/(b-a) for a ≤ x ≤ b

E[X] = ∫ₐᵇ x · (1/(b-a)) dx
     = (1/(b-a)) · [x²/2]ₐᵇ
     = (1/(b-a)) · (b²/2 - a²/2)
     = (b² - a²) / (2(b-a))
     = (b+a)(b-a) / (2(b-a))
     = (a + b) / 2

Midpoint of interval!
```

---

## 7. Variance and Standard Deviation

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

Often easier to compute!
```

### Properties

| Property | Formula |
|----------|---------|
| **Non-negative** | Var(X) ≥ 0 |
| **Constant** | Var(c) = 0 |
| **Scaling** | Var(aX + b) = a²Var(X) |
| **Sum (independent)** | Var(X + Y) = Var(X) + Var(Y) if independent |

### Example: Die Roll (continued)

```
X = fair die roll, E[X] = 3.5

E[X²] = 1²·(1/6) + 2²·(1/6) + 3²·(1/6) + 4²·(1/6) + 5²·(1/6) + 6²·(1/6)
      = (1 + 4 + 9 + 16 + 25 + 36)/6
      = 91/6
      = 15.167

Var(X) = E[X²] - (E[X])²
       = 15.167 - (3.5)²
       = 15.167 - 12.25
       = 2.917

SD(X) = √2.917 ≈ 1.708
```

### Example: Bernoulli(p)

```
X ~ Bernoulli(p)

E[X] = p

E[X²] = 0²·(1-p) + 1²·p = p

Var(X) = E[X²] - (E[X])²
       = p - p²
       = p(1 - p)

For fair coin (p = 0.5):
Var(X) = 0.5 × 0.5 = 0.25
SD(X) = 0.5
```

---

## 8. Moments

### Raw Moments

**k-th raw moment:**
```
E[Xᵏ] = Σ xᵏ·p(x)  or  ∫ xᵏ·f(x)dx
```

### Central Moments

**k-th central moment:**
```
E[(X - μ)ᵏ]
```

| Moment | Name | Meaning |
|--------|------|---------|
| **1st raw** | Mean (μ) | Center |
| **2nd central** | Variance (σ²) | Spread |
| **3rd standardized** | Skewness | Asymmetry |
| **4th standardized** | Kurtosis | Tail heaviness |

### Skewness

```
Skewness = E[(X - μ)³] / σ³

Skewness > 0: Right-skewed (long right tail)
Skewness < 0: Left-skewed (long left tail)
Skewness = 0: Symmetric
```

### Kurtosis

```
Kurtosis = E[(X - μ)⁴] / σ⁴

Kurtosis > 3: Heavy tails (leptokurtic)
Kurtosis = 3: Normal (mesokurtic)
Kurtosis < 3: Light tails (platykurtic)

Excess Kurtosis = Kurtosis - 3
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Discrete Random Variables ===

print("=" * 60)
print("DISCRETE RANDOM VARIABLES")
print("=" * 60)

# Fair die
x_die = np.array([1, 2, 3, 4, 5, 6])
p_die = np.array([1/6] * 6)

E_X = np.sum(x_die * p_die)
E_X2 = np.sum(x_die**2 * p_die)
Var_X = E_X2 - E_X**2

print(f"\nFair Die Roll:")
print(f"E[X] = {E_X:.4f}")
print(f"E[X²] = {E_X2:.4f}")
print(f"Var(X) = {Var_X:.4f}")
print(f"SD(X) = {np.sqrt(Var_X):.4f}")

# Biased coin (Bernoulli)
p = 0.7
x_coin = np.array([0, 1])
p_coin = np.array([1-p, p])

E_coin = np.sum(x_coin * p_coin)
Var_coin = np.sum((x_coin - E_coin)**2 * p_coin)

print(f"\nBiased Coin (p={p}):")
print(f"E[X] = {E_coin:.4f}")
print(f"Var(X) = {Var_coin:.4f}")
print(f"SD(X) = {np.sqrt(Var_coin):.4f}")

# === Continuous Random Variables ===

print("\n" + "=" * 60)
print("CONTINUOUS RANDOM VARIABLES")
print("=" * 60)

# Uniform distribution
a, b = 0, 10
uniform = stats.uniform(loc=a, scale=b-a)

print(f"\nUniform({a}, {b}):")
print(f"E[X] = {uniform.mean():.4f}")
print(f"Var(X) = {uniform.var():.4f}")
print(f"SD(X) = {uniform.std():.4f}")

# Normal distribution
mu, sigma = 100, 15
normal = stats.norm(loc=mu, scale=sigma)

print(f"\nNormal(μ={mu}, σ={sigma}):")
print(f"E[X] = {normal.mean():.4f}")
print(f"Var(X) = {normal.var():.4f}")
print(f"SD(X) = {normal.std():.4f}")

# Exponential distribution
lambda_param = 0.5
exponential = stats.expon(scale=1/lambda_param)

print(f"\nExponential(λ={lambda_param}):")
print(f"E[X] = {exponential.mean():.4f}")
print(f"Var(X) = {exponential.var():.4f}")

# === Visualization of Distributions ===

def plot_distribution(dist, name, x_range, n_samples=10000):
    """Plot PDF and histogram of distribution"""
    
    x = np.linspace(x_range[0], x_range[1], 400)
    pdf = dist.pdf(x)
    
    samples = dist.rvs(n_samples)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PDF plot
    ax1.plot(x, pdf, 'b-', linewidth=2, label='PDF')
    ax1.fill_between(x, pdf, alpha=0.3)
    ax1.axvline(dist.mean(), color='r', linestyle='--', label=f'Mean = {dist.mean():.2f}')
    ax1.set_title(f'{name} - PDF')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
    ax2.plot(x, pdf, 'r-', linewidth=2, label='Theoretical PDF')
    ax2.axvline(dist.mean(), color='g', linestyle='--', label=f'Mean = {dist.mean():.2f}')
    ax2.set_title(f'{name} - Histogram of {n_samples} samples')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot various distributions
plot_distribution(stats.norm(0, 1), 'Standard Normal', (-4, 4))
plot_distribution(stats.expon(scale=2), 'Exponential (λ=0.5)', (0, 15))
plot_distribution(stats.uniform(0, 10), 'Uniform(0, 10)', (0, 10))

# === Expected Value Simulation ===

def simulate_expected_value(dist, name, n_trials=10000):
    """Simulate convergence of sample mean to expected value"""
    
    true_mean = dist.mean()
    
    # Generate samples
    samples = dist.rvs(n_trials)
    
    # Running average
    running_means = np.cumsum(samples) / np.arange(1, n_trials + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_trials + 1), running_means, 'b-', linewidth=1, alpha=0.7)
    plt.axhline(y=true_mean, color='r', linestyle='--', linewidth=2, label=f'True Mean = {true_mean:.4f}')
    plt.title(f'Convergence of Sample Mean to E[X] - {name}')
    plt.xlabel('Number of trials')
    plt.ylabel('Running average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\n{name}:")
    print(f"True E[X] = {true_mean:.6f}")
    print(f"Sample mean after {n_trials} trials = {running_means[-1]:.6f}")
    print(f"Error = {abs(running_means[-1] - true_mean):.6f}")

# Test convergence
simulate_expected_value(stats.norm(0, 1), 'Standard Normal')
simulate_expected_value(stats.expon(scale=2), 'Exponential (λ=0.5)')

# === CDF Visualization ===

def plot_cdf_comparison(dist, name, x_range):
    """Plot PDF and CDF side by side"""
    
    x = np.linspace(x_range[0], x_range[1], 400)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PDF
    ax1.plot(x, pdf, 'b-', linewidth=2)
    ax1.fill_between(x, pdf, alpha=0.3)
    ax1.set_title(f'{name} - PDF')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, alpha=0.3)
    
    # CDF
    ax2.plot(x, cdf, 'g-', linewidth=2)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.set_title(f'{name} - CDF')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_cdf_comparison(stats.norm(0, 1), 'Standard Normal', (-4, 4))
plot_cdf_comparison(stats.expon(scale=1), 'Exponential', (0, 8))

# === Discrete Distribution Examples ===

def analyze_discrete_distribution(x_values, probabilities, name):
    """Analyze a discrete distribution"""
    
    x = np.array(x_values)
    p = np.array(probabilities)
    
    # Verify valid PMF
    assert np.all(p >= 0), "Probabilities must be non-negative"
    assert np.isclose(np.sum(p), 1), "Probabilities must sum to 1"
    
    # Calculate moments
    E_X = np.sum(x * p)
    E_X2 = np.sum(x**2 * p)
    Var_X = E_X2 - E_X**2
    
    print(f"\n{name}:")
    print(f"Values: {x}")
    print(f"Probabilities: {p}")
    print(f"E[X] = {E_X:.4f}")
    print(f"Var(X) = {Var_X:.4f}")
    print(f"SD(X) = {np.sqrt(Var_X):.4f}")
    
    # Plot PMF and CDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PMF
    ax1.bar(x, p, alpha=0.7, edgecolor='black')
    ax1.set_title(f'{name} - PMF')
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(X = x)')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # CDF
    cdf = np.cumsum(p)
    ax2.step(x, cdf, where='post', linewidth=2)
    ax2.scatter(x, cdf, s=100, zorder=5)
    ax2.set_title(f'{name} - CDF')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example: Custom discrete distribution
analyze_discrete_distribution(
    [1, 2, 3, 4, 5],
    [0.1, 0.15, 0.3, 0.25, 0.2],
    "Custom Distribution"
)

# Example: Binomial distribution
n, p = 10, 0.3
x_binom = np.arange(0, n+1)
p_binom = stats.binom.pmf(x_binom, n, p)
analyze_discrete_distribution(x_binom, p_binom, f"Binomial(n={n}, p={p})")
```

---

## 📊 Summary Table

| Concept | Discrete | Continuous |
|---------|----------|------------|
| **Probability function** | PMF: p(x) = P(X=x) | PDF: f(x) |
| **Normalization** | Σ p(x) = 1 | ∫ f(x) dx = 1 |
| **CDF** | F(x) = Σ_{t≤x} p(t) | F(x) = ∫_{-∞}^x f(t)dt |
| **Expected value** | E[X] = Σ x·p(x) | E[X] = ∫ x·f(x)dx |
| **Variance** | Var(X) = Σ (x-μ)²·p(x) | Var(X) = ∫ (x-μ)²·f(x)dx |
| **Computational formula** | Var(X) = E[X²] - (E[X])² | Same |

---

## 🎯 ML Applications

| Application | Random Variable Concept |
|-------------|------------------------|
| **Classification** | Output is discrete RV |
| **Regression** | Output is continuous RV |
| **Generative Models** | Model distribution of data RV |
| **Uncertainty** | Variance of predictions |
| **Loss Functions** | Expected loss minimization |
| **Bayesian ML** | Posterior distributions |

---

## ❓ Quick Check Questions

1. What's the difference between discrete and continuous RVs?
2. Why is P(X = x) = 0 for continuous RVs?
3. How do you compute E[X] for discrete vs continuous?
4. What is the computational formula for variance?
5. What does the CDF represent?
6. How are PDF and CDF related?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Discrete vs Continuous:**
   - Discrete: Countable values, PMF
   - Continuous: Interval of values, PDF

2. **P(X = x) = 0 for continuous:**
   - Probability is area under PDF
   - Single point has zero width
   - Only ranges have non-zero probability

3. **E[X] computation:**
   - Discrete: Σ x·p(x)
   - Continuous: ∫ x·f(x)dx

4. **Computational variance formula:**
   - Var(X) = E[X²] - (E[X])²
   - Often easier than definition

5. **CDF meaning:**
   - F(x) = P(X ≤ x)
   - Cumulative probability up to x

6. **PDF-CDF relationship:**
   - CDF is integral of PDF
   - PDF is derivative of CDF

</details>
---

**Status:** ✅ Complete
**Next:** Joint Distributions
