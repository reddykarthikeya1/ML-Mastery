# 1.3.5 Important Continuous Distributions

## 🎯 Quick Overview
- **Uniform**: Equal probability over interval
- **Normal**: Bell curve, most important distribution
- **Exponential**: Waiting times, memoryless
- **Foundation for**: Regression assumptions, neural network initialization, Bayesian priors

---

## 1. Uniform Distribution

### Definition

**Equal probability over interval [a, b]:**
```
X ~ Uniform(a, b)

PDF: f(x) = { 1/(b-a),  a ≤ x ≤ b
            { 0,        otherwise

CDF: F(x) = { 0,         x < a
            { (x-a)/(b-a), a ≤ x ≤ b
            { 1,         x > b
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| a | Minimum | -∞ < a < b |
| b | Maximum | a < b < ∞ |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = (a+b)/2 |
| **Variance** | Var(X) = (b-a)²/12 |
| **Skewness** | 0 (symmetric) |
| **Kurtosis** | 1.8 (light tails) |

### Examples

```
X ~ Uniform(0, 1) - Standard uniform

f(x) = 1 for 0 ≤ x ≤ 1

P(0.3 < X < 0.7) = (0.7 - 0.3) / (1 - 0) = 0.7 - 0.3 = 0.4

E[X] = (0 + 1)/2 = 0.5
Var(X) = (1 - 0)²/12 = 1/12 ≈ 0.0833
```

### Python

```python
from scipy import stats

a, b = 0, 10
uniform = stats.uniform(loc=a, scale=b-a)

print(f"Mean = {uniform.mean()}")
print(f"Variance = {uniform.var()}")
print(f"P(3 < X < 7) = {uniform.cdf(7) - uniform.cdf(3)}")
```

---

## 2. Normal (Gaussian) Distribution

### Definition

**The most important continuous distribution:**
```
X ~ Normal(μ, σ²)

PDF: f(x) = (1/(σ√(2π))) · e^(-(x-μ)²/(2σ²))

Notation: X ~ N(μ, σ²)
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| μ | Mean (location) | -∞ < μ < ∞ |
| σ | Standard deviation (scale) | σ > 0 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = μ |
| **Variance** | Var(X) = σ² |
| **Skewness** | 0 (symmetric) |
| **Kurtosis** | 3 (mesokurtic) |
| **Excess Kurtosis** | 0 |

### Standard Normal

```
Z ~ N(0, 1) - Standard normal

PDF: φ(z) = (1/√(2π)) · e^(-z²/2)

CDF: Φ(z) = P(Z ≤ z)

Any normal can be standardized:
Z = (X - μ) / σ
```

### 68-95-99.7 Rule

```
For any normal distribution:

P(μ - σ < X < μ + σ) = 0.6827 ≈ 68%
P(μ - 2σ < X < μ + 2σ) = 0.9545 ≈ 95%
P(μ - 3σ < X < μ + 3σ) = 0.9973 ≈ 99.7%
```

### Examples

**Example 1: Heights**
```
Adult male heights: N(μ=175cm, σ=7cm)

P(X > 182) = P(Z > (182-175)/7)
           = P(Z > 1)
           = 1 - Φ(1)
           = 1 - 0.8413
           = 0.1587

About 16% are taller than 182cm

P(168 < X < 182) = P(-1 < Z < 1)
                 = Φ(1) - Φ(-1)
                 = 0.8413 - 0.1587
                 = 0.6826 ≈ 68%
```

**Example 2: Finding Percentiles**
```
Find 90th percentile of N(100, 15²):

z_0.90 = Φ^(-1)(0.90) = 1.28

x_0.90 = μ + z·σ = 100 + 1.28·15 = 119.2

90% of values are below 119.2
```

### Python

```python
from scipy import stats
import numpy as np

mu, sigma = 100, 15
normal = stats.norm(loc=mu, scale=sigma)

# PDF and CDF
print(f"P(X < 115) = {normal.cdf(115):.4f}")
print(f"P(X > 115) = {1 - normal.cdf(115):.4f}")
print(f"P(85 < X < 115) = {normal.cdf(115) - normal.cdf(85):.4f}")

# Percentiles
print(f"90th percentile: {normal.ppf(0.90):.2f}")
print(f"95th percentile: {normal.ppf(0.95):.2f}")

# Standardize
x = 115
z = (x - mu) / sigma
print(f"Z-score of {x}: {z:.2f}")
```

---

## 3. Exponential Distribution

### Definition

**Time between events in Poisson process:**
```
X ~ Exponential(λ)

PDF: f(x) = { λ·e^(-λx),  x ≥ 0
            { 0,          x < 0

CDF: F(x) = { 1 - e^(-λx),  x ≥ 0
            { 0,            x < 0
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| λ | Rate parameter | λ > 0 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = 1/λ |
| **Variance** | Var(X) = 1/λ² |
| **Skewness** | 2 |
| **Kurtosis** | 9 (heavy tails) |

### Memoryless Property

```
P(X > s + t | X > s) = P(X > t)

"Remaining time doesn't depend on elapsed time"

Example: If component survived 100 hours,
P(survive 50 more hours) = P(survive 50 hours from start)
```

### Relationship to Poisson

```
If events follow Poisson(λ) per unit time:

Time between events ~ Exponential(λ)

Number of events in time t ~ Poisson(λt)
```

### Examples

**Example 3: Component Lifetime**
```
Mean lifetime = 1000 hours, so λ = 1/1000 = 0.001

P(X > 1000) = e^(-0.001·1000) = e^(-1) = 0.368

P(X < 500) = 1 - e^(-0.001·500) = 1 - e^(-0.5) = 0.393

P(500 < X < 1500) = e^(-0.5) - e^(-1.5) = 0.607 - 0.223 = 0.384
```

### Python

```python
from scipy import stats

λ = 0.001  # rate
exponential = stats.expon(scale=1/λ)  # scale = 1/λ

print(f"Mean = {exponential.mean()}")
print(f"P(X > 1000) = {1 - exponential.cdf(1000):.4f}")
print(f"90th percentile = {exponential.ppf(0.90):.2f}")
```

---

## 4. Gamma Distribution

### Definition

**Sum of exponential waiting times:**
```
X ~ Gamma(α, β)

PDF: f(x) = (β^α / Γ(α)) · x^(α-1) · e^(-βx)

for x > 0, α > 0, β > 0

where Γ(α) is the gamma function
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| α | Shape | α > 0 |
| β | Rate | β > 0 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = α/β |
| **Variance** | Var(X) = α/β² |
| **Skewness** | 2/√α |

### Special Cases

```
Exponential(λ) = Gamma(α=1, β=λ)

Chi-squared(k) = Gamma(α=k/2, β=1/2)

Erlang(k, λ) = Gamma(α=k, β=λ)  [k is integer]
```

### Examples

```
Time until 3rd event, each event ~ Exponential(λ=2):

X ~ Gamma(α=3, β=2)

E[X] = 3/2 = 1.5
Var(X) = 3/4 = 0.75
```

---

## 5. Beta Distribution

### Definition

**Distribution on [0, 1] - perfect for probabilities:**
```
X ~ Beta(α, β)

PDF: f(x) = (1/B(α,β)) · x^(α-1) · (1-x)^(β-1)

for 0 < x < 1

where B(α,β) is the beta function
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| α | Shape parameter 1 | α > 0 |
| β | Shape parameter 2 | β > 0 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = α/(α+β) |
| **Variance** | Var(X) = αβ/((α+β)²(α+β+1)) |

### Special Cases

```
Beta(1, 1) = Uniform(0, 1)

Beta(0.5, 0.5) = Arcsine distribution (U-shaped)

As α, β → ∞ with α/(α+β) fixed → Normal
```

### Bayesian Interpretation

```
Prior for probability p:
p ~ Beta(α, β)

After observing k successes in n trials:
Posterior: p ~ Beta(α+k, β+n-k)

Conjugate prior for Bernoulli/Binomial!
```

### Examples

```
Prior belief about conversion rate:
p ~ Beta(α=10, β=90)  # Belief: around 10%

E[p] = 10/100 = 0.10

After 50 visitors, 8 conversions:
Posterior: p ~ Beta(10+8, 90+42) = Beta(18, 132)

E[p|data] = 18/150 = 0.12
```

---

## 6. Chi-Squared Distribution

### Definition

**Sum of squared standard normals:**
```
X ~ Chi-squared(k)

If Z₁, Z₂, ..., Zₖ ~ N(0,1) independent:
X = Z₁² + Z₂² + ... + Zₖ² ~ Chi-squared(k)

k = degrees of freedom
```

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = k |
| **Variance** | Var(X) = 2k |
| **Skewness** | √(8/k) |

### Applications

- Hypothesis testing (chi-squared tests)
- Confidence intervals for variance
- Goodness of fit tests

---

## 7. Student's t-Distribution

### Definition

**Ratio of normal to sqrt of chi-squared:**
```
X ~ t(k)

If Z ~ N(0,1) and V ~ Chi-squared(k):
X = Z / √(V/k) ~ t(k)

k = degrees of freedom
```

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = 0 (for k > 1) |
| **Variance** | Var(X) = k/(k-2) (for k > 2) |

### Properties

- Symmetric, bell-shaped like normal
- Heavier tails than normal
- As k → ∞, t(k) → N(0, 1)

### Applications

- t-tests for means
- Confidence intervals with unknown variance
- Small sample inference

---

## 8. F-Distribution

### Definition

**Ratio of two chi-squared variables:**
```
X ~ F(d₁, d₂)

If U ~ Chi-squared(d₁) and V ~ Chi-squared(d₂):
X = (U/d₁) / (V/d₂) ~ F(d₁, d₂)
```

### Applications

- ANOVA (analysis of variance)
- F-tests for comparing variances
- Regression model comparison

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Compare Continuous Distributions ===

def compare_continuous_distributions():
    """Compare different continuous distributions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.linspace(-4, 10, 400)
    
    # 1. Normal distributions
    ax = axes[0, 0]
    for mu, sigma, style in [(0, 1, '-'), (0, 2, '--'), (2, 1, '-.')]:
        dist = stats.norm(mu, sigma)
        ax.plot(x, dist.pdf(x), style, linewidth=2, label=f'μ={mu}, σ={sigma}')
    ax.set_title('Normal Distributions')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Exponential distributions
    ax = axes[0, 1]
    x_exp = np.linspace(0, 10, 400)
    for λ, style in [(0.5, '-'), (1, '--'), (2, '-.')]:
        dist = stats.expon(scale=1/λ)
        ax.plot(x_exp, dist.pdf(x_exp), style, linewidth=2, label=f'λ={λ}')
    ax.set_title('Exponential Distributions')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Beta distributions
    ax = axes[1, 0]
    x_beta = np.linspace(0, 1, 400)
    for α, β, style in [(2, 5, '-'), (5, 2, '--'), (2, 2, '-.')]:
        dist = stats.beta(α, β)
        ax.plot(x_beta, dist.pdf(x_beta), style, linewidth=2, label=f'α={α}, β={β}')
    ax.set_title('Beta Distributions')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Gamma distributions
    ax = axes[1, 1]
    x_gamma = np.linspace(0, 15, 400)
    for α, β, style in [(1, 1, '-'), (2, 1, '--'), (5, 1, '-.')]:
        dist = stats.gamma(α, scale=1/β)
        ax.plot(x_gamma, dist.pdf(x_gamma), style, linewidth=2, label=f'α={α}, β={β}')
    ax.set_title('Gamma Distributions')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_continuous_distributions()

# === Normal Distribution Properties ===

def normal_properties():
    """Demonstrate 68-95-99.7 rule"""
    
    mu, sigma = 100, 15
    normal = stats.norm(mu, sigma)
    
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, normal.pdf(x), 'b-', linewidth=2)
    plt.fill_between(x, normal.pdf(x), alpha=0.3)
    
    # 68%
    plt.axvspan(mu - sigma, mu + sigma, alpha=0.3, color='green', label='68%')
    # 95%
    plt.axvspan(mu - 2*sigma, mu + 2*sigma, alpha=0.3, color='orange', label='95%')
    # 99.7%
    plt.axvspan(mu - 3*sigma, mu + 3*sigma, alpha=0.3, color='red', label='99.7%')
    
    plt.axvline(mu, color='black', linestyle='--', linewidth=2)
    plt.title(f'Normal Distribution N(μ={mu}, σ={sigma}) - 68-95-99.7 Rule')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print exact probabilities
    print("68-95-99.7 Rule:")
    print(f"P(μ ± σ) = {normal.cdf(mu + sigma) - normal.cdf(mu - sigma):.6f}")
    print(f"P(μ ± 2σ) = {normal.cdf(mu + 2*sigma) - normal.cdf(mu - 2*sigma):.6f}")
    print(f"P(μ ± 3σ) = {normal.cdf(mu + 3*sigma) - normal.cdf(mu - 3*sigma):.6f}")

normal_properties()

# === Central Limit Theorem Demo ===

def central_limit_theorem_demo():
    """Demonstrate CLT with different source distributions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    np.random.seed(42)
    n_simulations = 10000
    
    # Different source distributions
    distributions = [
        (lambda: np.random.uniform(0, 1, n_simulations), 'Uniform(0,1)'),
        (lambda: np.random.exponential(1, n_simulations), 'Exponential(1)'),
        (lambda: np.random.poisson(2, n_simulations), 'Poisson(2)'),
        (lambda: np.random.binomial(10, 0.3, n_simulations), 'Binomial(10, 0.3)'),
    ]
    
    for ax, (generator, name) in zip(axes.flatten(), distributions):
        # Sample means for different sample sizes
        for n in [1, 5, 30]:
            sample_means = [np.mean(generator()) for _ in range(n_simulations // n)]
            
            ax.hist(sample_means, bins=50, density=True, alpha=0.3, 
                   label=f'n={n}', range=(-2, 8))
        
        ax.set_title(f'Sample Means from {name}')
        ax.set_xlabel('Sample mean')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

central_limit_theorem_demo()

# === t-distribution vs Normal ===

def t_vs_normal():
    """Compare t-distribution to normal"""
    
    x = np.linspace(-4, 4, 400)
    
    plt.figure(figsize=(12, 6))
    
    # Standard normal
    normal = stats.norm(0, 1)
    plt.plot(x, normal.pdf(x), 'k-', linewidth=3, label='Normal(0,1)')
    
    # t-distributions with different df
    for df in [1, 3, 5, 10, 30]:
        t_dist = stats.t(df)
        plt.plot(x, t_dist.pdf(x), '-', linewidth=2, label=f't({df})')
    
    plt.title('t-Distribution vs Standard Normal')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print tail probabilities
    print("\nTail Probabilities P(|X| > 2):")
    print(f"Normal: {2 * (1 - normal.cdf(2)):.6f}")
    for df in [1, 3, 5, 10, 30]:
        t_dist = stats.t(df)
        print(f"t({df}): {2 * (1 - t_dist.cdf(2)):.6f}")

t_vs_normal()

# === Bayesian Updating with Beta ===

def beta_bayesian_updating():
    """Demonstrate Bayesian updating with Beta-Binomial"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.linspace(0, 1, 400)
    
    # Prior
    α_prior, β_prior = 10, 90
    prior = stats.beta(α_prior, β_prior)
    
    # Data: 50 visitors, 8 conversions
    successes, failures = 8, 42
    
    # Posterior
    α_post = α_prior + successes
    β_post = β_prior + failures
    posterior = stats.beta(α_post, β_post)
    
    # Plot prior
    ax = axes[0, 0]
    ax.plot(x, prior.pdf(x), 'b-', linewidth=2)
    ax.fill_between(x, prior.pdf(x), alpha=0.3)
    ax.axvline(prior.mean(), color='red', linestyle='--', label=f'Mean = {prior.mean():.3f}')
    ax.set_title(f'Prior: Beta({α_prior}, {β_prior})')
    ax.set_xlabel('p')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot likelihood
    ax = axes[0, 1]
    likelihood = stats.binom.pmf(successes, successes + failures, x)
    ax.plot(x, likelihood / likelihood.max(), 'g-', linewidth=2)
    ax.set_title(f'Likelihood: {successes} successes in {successes + failures} trials')
    ax.set_xlabel('p')
    ax.set_ylabel('Normalized likelihood')
    ax.grid(True, alpha=0.3)
    
    # Plot posterior
    ax = axes[1, 0]
    ax.plot(x, posterior.pdf(x), 'r-', linewidth=2)
    ax.fill_between(x, posterior.pdf(x), alpha=0.3)
    ax.axvline(posterior.mean(), color='blue', linestyle='--', label=f'Mean = {posterior.mean():.3f}')
    ax.set_title(f'Posterior: Beta({α_post}, {β_post})')
    ax.set_xlabel('p')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Compare prior and posterior
    ax = axes[1, 1]
    ax.plot(x, prior.pdf(x), 'b--', linewidth=2, label='Prior', alpha=0.7)
    ax.plot(x, posterior.pdf(x), 'r-', linewidth=2, label='Posterior', alpha=0.7)
    ax.axvline(prior.mean(), color='blue', linestyle=':', linewidth=2)
    ax.axvline(posterior.mean(), color='red', linestyle=':', linewidth=2)
    ax.set_title('Prior vs Posterior')
    ax.set_xlabel('p')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Bayesian Updating for Conversion Rate:")
    print(f"Prior: Beta({α_prior}, {β_prior}), Mean = {prior.mean():.4f}")
    print(f"Data: {successes} successes in {successes + failures} trials")
    print(f"Posterior: Beta({α_post}, {β_post}), Mean = {posterior.mean():.4f}")
    print(f"95% Credible Interval: [{posterior.ppf(0.025):.4f}, {posterior.ppf(0.975):.4f}]")

beta_bayesian_updating()
```

---

## 📊 Summary Table

| Distribution | PDF | Mean | Variance | Use Case |
|--------------|-----|------|----------|----------|
| **Uniform(a,b)** | 1/(b-a) | (a+b)/2 | (b-a)²/12 | Equal likelihood |
| **Normal(μ,σ²)** | (1/σ√(2π))e^(-(x-μ)²/(2σ²)) | μ | σ² | Default, CLT |
| **Exponential(λ)** | λe^(-λx) | 1/λ | 1/λ² | Waiting time |
| **Gamma(α,β)** | (β^α/Γ(α))x^(α-1)e^(-βx) | α/β | α/β² | Sum of exponentials |
| **Beta(α,β)** | (1/B(α,β))x^(α-1)(1-x)^(β-1) | α/(α+β) | complex | Probabilities, priors |
| **Chi-squared(k)** | complex | k | 2k | Hypothesis tests |
| **t(k)** | complex | 0 | k/(k-2) | Small samples |
| **F(d₁,d₂)** | complex | d₂/(d₂-2) | complex | ANOVA, variance ratio |

---

## 🎯 ML Applications

| Application | Distribution |
|-------------|--------------|
| **Weight Initialization** | Normal, Uniform |
| **Dropout** | Bernoulli (mask) |
| **Bayesian Neural Nets** | Normal priors |
| **Variational Autoencoders** | Normal (latent space) |
| **Uncertainty Estimation** | t-distribution (heavy tails) |
| **Conversion Rate Modeling** | Beta-Binomial |

---

**Status:** ✅ Complete
**Next:** Joint Distributions
