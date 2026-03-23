# 1.3.7 Limit Theorems

## 🎯 Quick Overview
- **LLN**: Sample mean converges to population mean
- **CLT**: Sample means are normally distributed
- **Chebyshev**: Bounds for any distribution
- **Foundation for**: Statistical inference, confidence intervals, hypothesis testing

---

## 1. Chebyshev's Inequality

### Statement

**For ANY random variable with finite mean μ and variance σ²:**
```
P(|X - μ| ≥ kσ) ≤ 1/k²

Or equivalently:
P(|X - μ| < kσ) ≥ 1 - 1/k²
```

### Interpretation

```
At least (1 - 1/k²) of values lie within k standard deviations of mean

k = 2: At least 75% within 2σ
k = 3: At least 89% within 3σ
k = 4: At least 94% within 4σ
```

### Examples

```
Any distribution with mean 100, SD 15:

P(70 < X < 130) = P(|X - 100| < 2·15) ≥ 0.75

At least 75% of values are between 70 and 130
(For normal: actually 95%)
```

---

## 2. Law of Large Numbers (LLN)

### Weak LLN

**As sample size increases, sample mean converges in probability:**
```
X̄ₙ → μ  as n → ∞

Formally: For any ε > 0,
P(|X̄ₙ - μ| ≥ ε) → 0  as n → ∞
```

### Strong LLN

**Almost sure convergence:**
```
P(lim(n→∞) X̄ₙ = μ) = 1
```

### Interpretation

```
Average of many independent trials approaches expected value

Example: Coin flips
- 10 flips: might get 7/10 = 70% heads
- 1000 flips: likely close to 50%
- 100000 flips: very close to 50%
```

---

## 3. Central Limit Theorem (CLT) ⭐ CRITICAL

### Statement

**For iid random variables with mean μ and variance σ²:**
```
X₁, X₂, ..., Xₙ iid with E[Xᵢ] = μ, Var(Xᵢ) = σ²

As n → ∞:
√n(X̄ₙ - μ)/σ → N(0, 1)

Or equivalently:
X̄ₙ ≈ N(μ, σ²/n)  for large n
```

### Key Points

1. **Works for ANY distribution** (with finite mean, variance)
2. **n ≥ 30** usually sufficient for good approximation
3. **More skewed distributions** need larger n

### Applications

```
1. Confidence intervals for means
2. Hypothesis testing
3. Quality control
4. A/B testing analysis
5. Monte Carlo methods
```

### Examples

**Example 1: Sample Mean**
```
Population: Exponential(λ=1), so μ=1, σ=1

Sample n=100:
X̄ ~ N(1, 1/100) = N(1, 0.01)

P(X̄ > 1.1) = P(Z > (1.1-1)/0.1) = P(Z > 1) = 0.1587
```

**Example 2: Sum of Random Variables**
```
Sum of 50 die rolls:

Each roll: μ = 3.5, σ² = 2.92

Sum: E[S] = 50·3.5 = 175
     Var(S) = 50·2.92 = 146
     SD(S) = √146 = 12.08

S ≈ N(175, 146)

P(S > 190) = P(Z > (190-175)/12.08) = P(Z > 1.24) = 0.1075
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Chebyshev's Inequality Demo ===

def chebyshev_demo():
    """Demonstrate Chebyshev's inequality"""
    
    distributions = [
        ('Normal(0,1)', stats.norm(0, 1)),
        ('Exponential(1)', stats.expon(1)),
        ('Uniform(-3,3)', stats.uniform(-3, 6)),
        ('Poisson(5)', stats.poisson(5)),
    ]
    
    print("Chebyshev's Inequality: P(|X-μ| < kσ) ≥ 1 - 1/k²")
    print("=" * 70)
    print(f"{'Distribution':<20} {'k=2 (≥75%)':<15} {'k=3 (≥89%)':<15} {'k=4 (≥94%)':<15}")
    print("-" * 70)
    
    for name, dist in distributions:
        samples = dist.rvs(100000)
        mean = np.mean(samples)
        std = np.std(samples)
        
        for k in [2, 3, 4]:
            proportion = np.mean(np.abs(samples - mean) < k * std)
            if k == 2:
                print(f"{name:<20} {proportion:.4f}", end="")
            elif k == 3:
                print(f"         {proportion:.4f}", end="")
            else:
                print(f"         {proportion:.4f}")

chebyshev_demo()

# === Law of Large Numbers Demo ===

def lln_demo():
    """Demonstrate Law of Large Numbers"""
    
    np.random.seed(42)
    
    # Different distributions
    distributions = [
        ('Bernoulli(0.5)', lambda n: np.random.binomial(1, 0.5, n)),
        ('Uniform(0,1)', lambda n: np.random.uniform(0, 1, n)),
        ('Exponential(1)', lambda n: np.random.exponential(1, n)),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, generator) in zip(axes, distributions):
        n_max = 10000
        samples = generator(n_max)
        
        # Running average
        running_means = np.cumsum(samples) / np.arange(1, n_max + 1)
        
        # True mean
        if 'Bernoulli' in name:
            true_mean = 0.5
        elif 'Uniform' in name:
            true_mean = 0.5
        else:
            true_mean = 1.0
        
        ax.plot(range(1, n_max + 1), running_means, 'b-', linewidth=1, alpha=0.7)
        ax.axhline(y=true_mean, color='r', linestyle='--', linewidth=2, label=f'True mean = {true_mean}')
        ax.set_title(f'LLN: {name}')
        ax.set_xlabel('Number of trials (n)')
        ax.set_ylabel('Running average')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_max)
    
    plt.tight_layout()
    plt.show()

lln_demo()

# === Central Limit Theorem Demo ===

def clt_demo():
    """Comprehensive CLT demonstration"""
    
    np.random.seed(42)
    n_simulations = 10000
    
    # Different source distributions
    distributions = [
        ('Uniform(0,1)', lambda n: np.random.uniform(0, 1, n), 0.5, 1/12),
        ('Exponential(1)', lambda n: np.random.exponential(1, n), 1, 1),
        ('Poisson(3)', lambda n: np.random.poisson(3, n), 3, 3),
        ('Binomial(10,0.3)', lambda n: np.random.binomial(10, 0.3, n), 3, 2.1),
    ]
    
    sample_sizes = [1, 5, 20, 50]
    
    fig, axes = plt.subplots(len(distributions), len(sample_sizes), figsize=(20, 15))
    
    for i, (name, generator, true_mean, true_var) in enumerate(distributions):
        for j, n in enumerate(sample_sizes):
            # Generate sample means
            sample_means = [np.mean(generator(n)) for _ in range(n_simulations)]
            
            ax = axes[i, j] if len(distributions) > 1 else axes[j]
            
            # Histogram
            ax.hist(sample_means, bins=50, density=True, alpha=0.6, label=f'n={n}')
            
            # Normal approximation
            approx_mean = true_mean
            approx_std = np.sqrt(true_var / n)
            x = np.linspace(min(sample_means), max(sample_means), 100)
            ax.plot(x, stats.norm.pdf(x, approx_mean, approx_std), 'r-', linewidth=2)
            
            if j == 0:
                ax.set_ylabel('Density')
            if i == len(distributions) - 1:
                ax.set_xlabel('Sample mean')
            
            ax.set_title(f'{name}\nn={n}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Central Limit Theorem: Sample Means Approach Normal Distribution', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Quantitative comparison
    print("\nCLT Verification (n=50):")
    print("=" * 70)
    print(f"{'Distribution':<25} {'Sample Mean':<15} {'True Mean':<15} {'Difference'}")
    print("-" * 70)
    
    for name, generator, true_mean, true_var in distributions:
        sample_means = [np.mean(generator(50)) for _ in range(n_simulations)]
        observed_mean = np.mean(sample_means)
        print(f"{name:<25} {observed_mean:<15.6f} {true_mean:<15.6f} {abs(observed_mean - true_mean):.6f}")

clt_demo()

# === CLT Application: Confidence Intervals ===

def clt_confidence_intervals():
    """Use CLT to create confidence intervals"""
    
    np.random.seed(42)
    
    # True population
    true_mean = 100
    true_std = 15
    
    # Take multiple samples
    n_samples = 100
    sample_size = 50
    
    confidence_levels = [0.90, 0.95, 0.99]
    
    print(f"\nConfidence Interval Simulation")
    print(f"True mean: {true_mean}, True SD: {true_std}")
    print(f"Sample size: {sample_size}, Number of samples: {n_samples}")
    print("=" * 70)
    
    for conf in confidence_levels:
        z = stats.norm.ppf(1 - (1-conf)/2)
        
        contains_true_mean = 0
        
        for _ in range(n_samples):
            sample = np.random.normal(true_mean, true_std, sample_size)
            sample_mean = np.mean(sample)
            se = true_std / np.sqrt(sample_size)
            
            ci_lower = sample_mean - z * se
            ci_upper = sample_mean + z * se
            
            if ci_lower <= true_mean <= ci_upper:
                contains_true_mean += 1
        
        coverage = contains_true_mean / n_samples
        print(f"{conf*100:.0f}% CI: Contains true mean {coverage:.2%} of time (expected: {conf:.2%})")

clt_confidence_intervals()
```

---

## 📊 Summary Table

| Theorem | Statement | Application |
|---------|-----------|-------------|
| **Chebyshev** | P(\|X-μ\| ≥ kσ) ≤ 1/k² | Universal bounds |
| **LLN** | X̄ₙ → μ as n → ∞ | Convergence of averages |
| **CLT** | √n(X̄ₙ-μ)/σ → N(0,1) | Normal approximation |

---

## 🎯 ML Applications

| Application | Limit Theorem |
|-------------|---------------|
| **Mini-batch Gradient Descent** | LLN (gradient estimates) |
| **Uncertainty Quantification** | CLT (confidence intervals) |
| **A/B Testing** | CLT (test statistics) |
| **Bootstrap Methods** | CLT (resampling) |
| **Monte Carlo Methods** | LLN (convergence) |

---

**Status:** ✅ Complete
**Next:** Estimation Theory
