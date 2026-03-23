# 1.3.9 Sampling Distributions

## 🎯 Quick Overview
- **Sampling distribution**: Distribution of a statistic over many samples
- **Standard error**: Standard deviation of sampling distribution
- **Foundation for**: Confidence intervals, hypothesis testing

---

## 1. Sampling Distribution of Sample Mean

### Definition

If we take many samples of size n from a population:
```
X̄₁, X̄₂, ..., X̄ₖ

The distribution of these sample means is the sampling distribution
```

### Properties

**If population is Normal(μ, σ²):**
```
X̄ ~ N(μ, σ²/n)

Mean: E[X̄] = μ
Variance: Var(X̄) = σ²/n
Std Error: SE(X̄) = σ/√n
```

**Central Limit Theorem (any population):**
```
As n → ∞:
X̄ → N(μ, σ²/n)

Rule of thumb: n ≥ 30 for good approximation
```

### Standard Error

```
SE(X̄) = σ/√n  (if σ known)
SE(X̄) = S/√n  (if σ unknown, use sample S)
```

**Interpretation:**
- Measures precision of sample mean
- Decreases as sample size increases
- Halving SE requires 4× sample size

---

## 2. Sampling Distribution of Sample Proportion

### Definition

For binary data (success/failure):
```
p̂ = X/n

where X = number of successes
```

### Properties

```
If n is large:
p̂ ~ N(p, p(1-p)/n)

Mean: E[p̂] = p
Variance: Var(p̂) = p(1-p)/n
Std Error: SE(p̂) = √(p(1-p)/n)
```

### Conditions for Normal Approximation

```
np ≥ 10 AND n(1-p) ≥ 10

Rule ensures enough successes and failures
```

---

## 3. Sampling Distribution of Sample Variance

### Chi-Squared Distribution

**If population is normal:**
```
(n-1)S²/σ² ~ χ²(n-1)

Degrees of freedom: df = n-1
```

### Properties of Chi-Squared

| Property | Value |
|----------|-------|
| Mean | df |
| Variance | 2×df |
| Shape | Right-skewed (approaches normal as df increases) |

---

## 4. t-Distribution in Sampling

### When σ Unknown

**Use sample standard deviation S:**
```
t = (X̄ - μ) / (S/√n) ~ t(n-1)

Degrees of freedom: df = n-1
```

### Properties of t-Distribution

| Property | Value |
|----------|-------|
| Shape | Bell-shaped, symmetric |
| Mean | 0 |
| Variance | df/(df-2) for df > 2 |
| Tails | Heavier than normal |
| As df → ∞ | Approaches N(0,1) |

### When to Use t vs z

| Situation | Distribution |
|-----------|--------------|
| σ known | z (normal) |
| σ unknown, n small (< 30) | t |
| σ unknown, n large (≥ 30) | z or t |

---

## 5. Chi-Squared Distribution in Sampling

### Applications

1. **Confidence intervals for variance**
2. **Goodness of fit tests**
3. **Test of independence**

### Confidence Interval for Variance

```
(n-1)S²/χ²_{α/2} ≤ σ² ≤ (n-1)S²/χ²_{1-α/2}

where χ² values from chi-squared table with df = n-1
```

---

## 6. F-Distribution in Sampling

### Definition

**Ratio of two chi-squared variables:**
```
F = (χ²₁/df₁) / (χ²₂/df₂) ~ F(df₁, df₂)
```

### Applications

1. **Comparing two variances**
2. **ANOVA (analysis of variance)**
3. **Regression model comparison**

### F-Test for Equal Variances

```
H₀: σ₁² = σ₂²
H₁: σ₁² ≠ σ₂²

Test statistic: F = S₁²/S₂²

Reject H₀ if F is extreme (use F-table)
```

---

## 7. Standard Error

### Definition

```
SE = Standard deviation of sampling distribution

Measures precision of statistic as estimator
```

### Common Standard Errors

| Statistic | Standard Error |
|-----------|---------------|
| Mean | σ/√n or S/√n |
| Proportion | √(p(1-p)/n) |
| Variance | σ²√(2/(n-1)) |
| Difference of means | √(σ₁²/n₁ + σ₂²/n₂) |

### Interpretation

```
Smaller SE → More precise estimate

SE decreases as √n:
- 4× sample size → halve SE
- 100× sample size → 10× smaller SE
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Sampling Distribution of Mean ===

np.random.seed(42)

# Population
population = stats.expon(scale=10)  # Exponential, mean=10

# Take many samples
n_samples = 10000
sample_size = 50

sample_means = [np.mean(population.rvs(sample_size)) 
                for _ in range(n_samples)]

# Plot sampling distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Population distribution
axes[0].hist(population.rvs(10000), bins=50, density=True, alpha=0.6)
axes[0].axvline(population.mean(), color='red', linestyle='--', 
                label=f'Population mean = {population.mean():.2f}')
axes[0].set_title('Population Distribution (Exponential)')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Sampling distribution of mean
axes[1].hist(sample_means, bins=50, density=True, alpha=0.6, label='Sample means')
axes[1].axvline(np.mean(sample_means), color='red', linestyle='--', 
                label=f'Mean of means = {np.mean(sample_means):.2f}')

# Normal approximation
x = np.linspace(min(sample_means), max(sample_means), 100)
approx_mean = population.mean()
approx_std = population.std() / np.sqrt(sample_size)
axes[1].plot(x, stats.norm.pdf(x, approx_mean, approx_std), 'g-', linewidth=2, 
             label=f'Normal approx\nN({approx_mean:.2f}, {approx_std:.2f})')

axes[1].set_title(f'Sampling Distribution of Mean (n={sample_size})')
axes[1].set_xlabel('Sample mean')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Standard Error Demonstration ===

print("Standard Error Demonstration")
print("=" * 50)

sample_sizes = [10, 25, 50, 100, 200, 500]
true_std = 15

print(f"{'Sample Size':<15} {'SE':<15} {'SE/SE(n=10)}'}")
print("-" * 45)

for n in sample_sizes:
    SE = true_std / np.sqrt(n)
    ratio = SE / (true_std / np.sqrt(10))
    print(f"{n:<15} {SE:<15.4f} {ratio:.4f}")

# === t-Distribution vs Normal ===

def t_vs_normal_sampling():
    """Compare t and normal for small samples"""
    
    np.random.seed(42)
    n = 10  # Small sample
    n_simulations = 10000
    
    t_stats = []
    z_stats = []
    
    true_mean = 100
    true_std = 15
    
    for _ in range(n_simulations):
        sample = np.random.normal(true_mean, true_std, n)
        X_bar = np.mean(sample)
        S = np.std(sample, ddof=1)
        
        # t-statistic (uses S)
        t = (X_bar - true_mean) / (S / np.sqrt(n))
        t_stats.append(t)
        
        # z-statistic (uses σ)
        z = (X_bar - true_mean) / (true_std / np.sqrt(n))
        z_stats.append(z)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # t-distribution
    axes[0].hist(t_stats, bins=50, density=True, alpha=0.6, label='Simulated t')
    x = np.linspace(-4, 4, 100)
    axes[0].plot(x, stats.t.pdf(x, df=n-1), 'r-', linewidth=2, label=f't({n-1})')
    axes[0].plot(x, stats.norm.pdf(x), 'b--', linewidth=2, label='Normal')
    axes[0].set_title(f't-Distribution (df={n-1})')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # z-distribution (normal)
    axes[1].hist(z_stats, bins=50, density=True, alpha=0.6, label='Simulated z')
    axes[1].plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='Normal(0,1)')
    axes[1].set_title('Standard Normal (z) Distribution')
    axes[1].set_xlabel('z')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

t_vs_normal_sampling()

# === Chi-Squared Distribution ===

def chi_squared_sampling():
    """Demonstrate chi-squared distribution for variance"""
    
    np.random.seed(42)
    n = 20
    n_simulations = 10000
    
    true_variance = 225  # 15²
    
    chi_squared_stats = []
    
    for _ in range(n_simulations):
        sample = np.random.normal(100, 15, n)
        S_squared = np.var(sample, ddof=1)
        
        # Chi-squared statistic
        chi_sq = (n - 1) * S_squared / true_variance
        chi_squared_stats.append(chi_sq)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.hist(chi_squared_stats, bins=50, density=True, alpha=0.6, label='Simulated')
    
    x = np.linspace(0, 40, 200)
    plt.plot(x, stats.chi2.pdf(x, df=n-1), 'r-', linewidth=2, 
             label=f'Chi-squared(df={n-1})')
    
    plt.title(f'Chi-Squared Distribution for Variance (df={n-1})')
    plt.xlabel('χ²')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Properties
    print(f"\nChi-Squared Distribution (df={n-1}):")
    print(f"Theoretical mean: {n-1}")
    print(f"Sample mean: {np.mean(chi_squared_stats):.2f}")
    print(f"Theoretical variance: {2*(n-1)}")
    print(f"Sample variance: {np.var(chi_squared_stats):.2f}")

chi_squared_sampling()

# === Confidence Interval Simulation ===

def ci_simulation():
    """Demonstrate confidence intervals"""
    
    np.random.seed(42)
    true_mean = 100
    true_std = 15
    n = 50
    n_sims = 100
    confidence = 0.95
    
    contains_true_mean = 0
    cis = []
    
    for _ in range(n_sims):
        sample = np.random.normal(true_mean, true_std, n)
        X_bar = np.mean(sample)
        SE = true_std / np.sqrt(n)
        
        z = stats.norm.ppf(1 - (1-confidence)/2)
        ci_lower = X_bar - z * SE
        ci_upper = X_bar + z * SE
        
        cis.append((ci_lower, ci_upper))
        
        if ci_lower <= true_mean <= ci_upper:
            contains_true_mean += 1
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    for i, (ci_lower, ci_upper) in enumerate(cis):
        color = 'green' if ci_lower <= true_mean <= ci_upper else 'red'
        plt.plot([ci_lower, ci_upper], [i, i], color=color, linewidth=2)
    
    plt.axvline(true_mean, color='black', linestyle='--', linewidth=2, 
                label=f'True mean = {true_mean}')
    plt.title(f'{confidence*100}% Confidence Intervals\n'
              f'Green: Contains true mean, Red: Does not\n'
              f'Coverage: {contains_true_mean/n_sims*100:.1f}%')
    plt.xlabel('Value')
    plt.ylabel('Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nConfidence Interval Simulation ({n_sims} samples):")
    print(f"Confidence level: {confidence*100}%")
    print(f"Intervals containing true mean: {contains_true_mean}/{n_sims}")
    print(f"Coverage: {contains_true_mean/n_sims*100:.1f}%")
    print(f"Expected coverage: {confidence*100}%")

ci_simulation()
```

---

## 📊 Summary Tables

### Sampling Distributions

| Statistic | Distribution | Parameters |
|-----------|--------------|------------|
| Mean (σ known) | Normal | N(μ, σ²/n) |
| Mean (σ unknown) | t | t(n-1) |
| Proportion | Normal | N(p, p(1-p)/n) |
| Variance | Chi-squared | χ²(n-1) |
| Ratio of variances | F | F(n₁-1, n₂-1) |

### Standard Errors

| Statistic | Formula |
|-----------|---------|
| Mean | S/√n |
| Proportion | √(p̂(1-p̂)/n) |
| Difference of means | √(S₁²/n₁ + S₂²/n₂) |
| Difference of proportions | √(p̂₁(1-p̂₁)/n₁ + p̂₂(1-p̂₂)/n₂) |

---

## 🎯 ML Applications

| Application | Sampling Distribution |
|-------------|----------------------|
| **Bootstrap** | Resampling distribution |
| **Cross-Validation** | Distribution of metrics |
| **A/B Testing** | Difference of means |
| **Uncertainty** | Standard errors |
| **Bayesian Methods** | Posterior distributions |

---

**Status:** ✅ Complete
**Next:** Bayesian Statistics
