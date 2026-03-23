# 1.3.13 Bayesian Statistics

## 🎯 Quick Overview
- **Bayesian inference**: Update beliefs with data
- **Prior → Posterior**: Learning from evidence
- **Conjugate priors**: Mathematical convenience
- **Foundation for**: Bayesian ML, uncertainty quantification, A/B testing

---

## 1. Bayes' Theorem Review

### Formula

```
P(H|D) = P(D|H) · P(H) / P(D)

Posterior = Likelihood × Prior / Evidence

Or:
P(H|D) ∝ P(D|H) × P(H)

Posterior ∝ Likelihood × Prior
```

### Terms

| Term | Meaning | Role |
|------|---------|------|
| **Posterior** P(H\|D) | Belief after seeing data | What we want |
| **Likelihood** P(D\|H) | Probability of data given hypothesis | From model |
| **Prior** P(H) | Belief before seeing data | Domain knowledge |
| **Evidence** P(D) | Marginal likelihood | Normalization |

---

## 2. Prior Selection

### Types of Priors

| Type | Description | Example |
|------|-------------|---------|
| **Informative** | Specific prior knowledge | Beta(10, 90) for 10% rate |
| **Weakly informative** | General constraints | Normal(0, 10) for effect size |
| **Non-informative** | Minimal assumptions | Uniform(0, 1) |
| **Jeffreys prior** | Invariant to transformation | Beta(0.5, 0.5) |

### Common Priors for Proportions

```
Beta(α, β) family:

Beta(1, 1) = Uniform(0, 1)         # Flat prior
Beta(0.5, 0.5) = Jeffreys prior    # Non-informative
Beta(10, 90)                       # Belief around 10%
Beta(100, 900)                     # Strong belief around 10%
```

### Common Priors for Continuous Parameters

```
Normal(μ₀, σ₀²) for means:
- Normal(0, 100)  # Weakly informative
- Normal(0, 1)    # Regularizing

Inverse-Gamma(α, β) for variances
Half-Cauchy(0, 5) for standard deviations
```

---

## 3. Likelihood Function

### Definition

```
L(θ|D) = P(D|θ)

Function of parameter given observed data
```

### Examples

**Bernoulli/Binomial:**
```
Data: k successes in n trials
L(p|k,n) = C(n,k) · pᵏ · (1-p)ⁿ⁻ᵏ
```

**Normal:**
```
Data: x₁, ..., xₙ
L(μ,σ²|x) = Π (1/√(2πσ²)) · exp(-(xᵢ-μ)²/(2σ²))
```

**Poisson:**
```
Data: k events
L(λ|k) = e^(-λ) · λᵏ / k!
```

---

## 4. Posterior Computation

### General Formula

```
P(θ|D) = P(D|θ) × P(θ) / P(D)

where P(D) = ∫ P(D|θ) × P(θ) dθ (normalization)
```

### Grid Approximation

```
1. Define grid of θ values
2. Compute prior for each θ
3. Compute likelihood for each θ
4. Multiply: unnormalized posterior
5. Normalize to sum to 1
```

### Analytical Solution (Conjugate Priors)

```
When prior and posterior are in same family:
- Beta-Binomial
- Gamma-Poisson
- Normal-Normal
```

---

## 5. Conjugate Priors

### Beta-Binomial

```
Prior:     p ~ Beta(α, β)
Likelihood: X|p ~ Binomial(n, p)
Posterior: p|X ~ Beta(α + X, β + n - X)

Example:
Prior: Beta(10, 90)  # Belief: 10% conversion
Data: 8 conversions from 50 visitors
Posterior: Beta(18, 132)  # Updated belief
```

### Gamma-Poisson

```
Prior:     λ ~ Gamma(α, β)
Likelihood: X|λ ~ Poisson(λ)
Posterior: λ|X ~ Gamma(α + X, β + 1)

Example:
Prior: Gamma(5, 1)  # Expect 5 events per interval
Data: 3 events observed
Posterior: Gamma(8, 2)
```

### Normal-Normal (Known Variance)

```
Prior:     μ ~ Normal(μ₀, σ₀²)
Likelihood: X|μ ~ Normal(μ, σ²/n)
Posterior: μ|X ~ Normal(μₙ, σₙ²)

where:
σₙ² = 1/(1/σ₀² + n/σ²)
μₙ = σₙ² × (μ₀/σ₀² + n·X̄/σ²)
```

---

## 6. Bayesian Inference

### Point Estimates

**Posterior Mean:**
```
E[θ|D] = ∫ θ · P(θ|D) dθ

Minimizes squared error loss
```

**Posterior Median:**
```
Value where P(θ < median|D) = 0.5

Minimizes absolute error loss
Robust to skewed posteriors
```

**MAP (Maximum A Posteriori):**
```
argmax P(θ|D)

Mode of posterior
Similar to MLE with regularization
```

### Credible Intervals

```
95% Credible Interval:
P(L ≤ θ ≤ U | D) = 0.95

Interpretation:
"There is 95% probability that θ is in this interval"

Different from frequentist CI!
```

### Highest Density Interval (HDI)

```
Shortest interval containing 95% of posterior
All points inside have higher probability than outside
```

---

## 7. Bayesian Hypothesis Testing

### Bayes Factor

```
BF₁₀ = P(D|H₁) / P(D|H₀)

Evidence for H₁ relative to H₀

Interpretation:
- BF > 1: Evidence for H₁
- BF < 1: Evidence for H₀
- BF = 1: No evidence either way
```

### Interpretation Scale

| Bayes Factor | Evidence |
|--------------|----------|
| 1-3 | Anecdotal |
| 3-10 | Moderate |
| 10-30 | Strong |
| 30-100 | Very strong |
| > 100 | Decisive |

---

## 8. MCMC Concepts

### When Analytical Solution Not Available

```
Use sampling methods:
- Metropolis-Hastings
- Gibbs sampling
- Hamiltonian Monte Carlo (HMC)
- No-U-Turn Sampler (NUTS)
```

### Metropolis-Hastings Algorithm

```
1. Start at current θ
2. Propose new θ' from proposal distribution
3. Calculate acceptance ratio:
   r = P(θ'|D) / P(θ|D)
4. Accept θ' with probability min(1, r)
5. Repeat
```

### Gibbs Sampling

```
Special case of Metropolis-Hastings
Sample each parameter from its conditional distribution
Always accepted
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta, gamma, norm

# === Beta-Binomial Conjugate ===

print("Beta-Binomial Bayesian Inference")
print("=" * 50)

# Prior: Beta(10, 90) - belief around 10% conversion
alpha_prior, beta_prior = 10, 90

# Data: 8 conversions from 50 visitors
conversions = 8
n_visitors = 50

# Posterior: Beta(α + X, β + n - X)
alpha_post = alpha_prior + conversions
beta_post = beta_prior + n_visitors - conversions

print(f"Prior: Beta({alpha_prior}, {beta_prior})")
print(f"Data: {conversions} conversions from {n_visitors} visitors")
print(f"Posterior: Beta({alpha_post}, {beta_post})")

# Prior and posterior means
prior_mean = alpha_prior / (alpha_prior + beta_prior)
post_mean = alpha_post / (alpha_post + beta_post)
mle = conversions / n_visitors

print(f"\nPrior mean: {prior_mean:.4f}")
print(f"Posterior mean: {post_mean:.4f}")
print(f"MLE (data only): {mle:.4f}")

# 95% Credible Interval
ci_lower, ci_upper = beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"\n95% Credible Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Plot
x = np.linspace(0, 0.5, 400)

plt.figure(figsize=(12, 6))
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), 'b--', linewidth=2, label='Prior')
plt.plot(x, beta.pdf(x, alpha_post, beta_post), 'r-', linewidth=2, label='Posterior')
plt.axvline(prior_mean, color='blue', linestyle=':', label=f'Prior mean = {prior_mean:.3f}')
plt.axvline(post_mean, color='red', linestyle='-', label=f'Posterior mean = {post_mean:.3f}')
plt.axvline(mle, color='green', linestyle='--', label=f'MLE = {mle:.3f}')
plt.fill_between(x, beta.pdf(x, alpha_post, beta_post), alpha=0.3, color='red')
plt.title('Beta-Binomial Bayesian Inference')
plt.xlabel('Conversion rate (p)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# === Gamma-Poisson Conjugate ===

print("\nGamma-Poisson Bayesian Inference")
print("=" * 50)

# Prior: Gamma(5, 1) - expect 5 events per interval
alpha_prior, beta_prior = 5, 1

# Data: 3 events observed
k_events = 3

# Posterior: Gamma(α + k, β + 1)
alpha_post = alpha_prior + k_events
beta_post = beta_prior + 1

print(f"Prior: Gamma({alpha_prior}, {beta_prior})")
print(f"Data: {k_events} events")
print(f"Posterior: Gamma({alpha_post}, {beta_post})")

prior_mean = alpha_prior / beta_prior
post_mean = alpha_post / beta_post
mle = k_events

print(f"\nPrior mean: {prior_mean:.2f}")
print(f"Posterior mean: {post_mean:.2f}")
print(f"MLE: {mle}")

# Plot
x = np.linspace(0, 15, 400)

plt.figure(figsize=(12, 6))
plt.plot(x, gamma.pdf(x, alpha_prior, scale=1/beta_prior), 'b--', linewidth=2, label='Prior')
plt.plot(x, gamma.pdf(x, alpha_post, scale=1/beta_post), 'r-', linewidth=2, label='Posterior')
plt.axvline(prior_mean, color='blue', linestyle=':', label=f'Prior mean = {prior_mean:.2f}')
plt.axvline(post_mean, color='red', linestyle='-', label=f'Posterior mean = {post_mean:.2f}')
plt.axvline(mle, color='green', linestyle='--', label=f'MLE = {mle}')
plt.fill_between(x, gamma.pdf(x, alpha_post, scale=1/beta_post), alpha=0.3, color='red')
plt.title('Gamma-Poisson Bayesian Inference')
plt.xlabel('Rate (λ)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# === Normal-Normal Conjugate ===

print("\nNormal-Normal Bayesian Inference")
print("=" * 50)

# Prior: Normal(μ₀=100, σ₀=15)
mu_prior, sigma_prior = 100, 15

# Data: sample of n=25 with X̄=105, σ=15 (known)
x_bar, sigma_known, n = 105, 15, 25

# Posterior precision (1/variance)
prior_precision = 1 / sigma_prior**2
data_precision = n / sigma_known**2
post_precision = prior_precision + data_precision

# Posterior parameters
sigma_post = np.sqrt(1 / post_precision)
mu_post = sigma_post**2 * (mu_prior / sigma_prior**2 + n * x_bar / sigma_known**2)

print(f"Prior: N({mu_prior}, {sigma_prior}²)")
print(f"Data: X̄={x_bar}, n={n}, σ={sigma_known}")
print(f"Posterior: N({mu_post:.2f}, {sigma_post:.2f}²)")

print(f"\nPrior mean: {mu_prior}")
print(f"Posterior mean: {mu_post:.2f}")
print(f"Sample mean: {x_bar}")

# Weight of prior vs data
prior_weight = prior_precision / post_precision
data_weight = data_precision / post_precision
print(f"\nPrior weight: {prior_weight:.2%}")
print(f"Data weight: {data_weight:.2%}")

# Plot
x = np.linspace(80, 130, 400)

plt.figure(figsize=(12, 6))
plt.plot(x, norm.pdf(x, mu_prior, sigma_prior), 'b--', linewidth=2, label='Prior')
plt.plot(x, norm.pdf(x, x_bar, sigma_known/np.sqrt(n)), 'g-.', linewidth=2, label='Likelihood')
plt.plot(x, norm.pdf(x, mu_post, sigma_post), 'r-', linewidth=2, label='Posterior')
plt.axvline(mu_prior, color='blue', linestyle=':', label=f'Prior mean = {mu_prior}')
plt.axvline(x_bar, color='green', linestyle=':', label=f'Sample mean = {x_bar}')
plt.axvline(mu_post, color='red', linestyle='-', label=f'Posterior mean = {mu_post:.2f}')
plt.fill_between(x, norm.pdf(x, mu_post, sigma_post), alpha=0.3, color='red')
plt.title('Normal-Normal Bayesian Inference')
plt.xlabel('μ')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# === Grid Approximation ===

def grid_approximation(prior_func, likelihood_func, theta_grid):
    """Compute posterior using grid approximation"""
    
    # Compute prior
    prior = prior_func(theta_grid)
    
    # Compute likelihood
    likelihood = likelihood_func(theta_grid)
    
    # Unnormalized posterior
    posterior_unnorm = prior * likelihood
    
    # Normalize
    posterior = posterior_unnorm / np.sum(posterior_unnorm)
    
    return posterior

# Example: Custom prior with binomial likelihood
print("\nGrid Approximation Example")
print("=" * 50)

theta = np.linspace(0, 1, 1000)

# Triangular prior (peaked at 0.5)
def prior_func(theta):
    return 1 - np.abs(2*theta - 1)

# Binomial likelihood
conversions, n_visitors = 30, 100
def likelihood_func(theta):
    return stats.binom.pmf(conversions, n_visitors, theta)

posterior = grid_approximation(prior_func, likelihood_func, theta)

# Find posterior statistics
post_mean = np.sum(theta * posterior)
ci_indices = np.argsort(np.cumsum(posterior))
ci_lower = theta[np.searchsorted(np.cumsum(posterior)[np.argsort(np.cumsum(posterior))], 0.025)]
ci_upper = theta[np.searchsorted(np.cumsum(posterior)[np.argsort(np.cumsum(posterior))], 0.975)]

print(f"Posterior mean: {post_mean:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(theta, prior_func(theta), 'b-', linewidth=2)
plt.title('Prior')
plt.xlabel('θ')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(theta, likelihood_func(theta), 'g-', linewidth=2)
plt.title('Likelihood')
plt.xlabel('θ')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(theta, posterior, 'r-', linewidth=2)
plt.axvline(post_mean, color='red', linestyle='--', label=f'Mean = {post_mean:.3f}')
plt.fill_between(theta, posterior, alpha=0.3)
plt.title('Posterior (Grid Approximation)')
plt.xlabel('θ')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Bayesian vs Frequentist Comparison ===

def bayesian_frequentist_comparison():
    """Compare Bayesian and Frequentist approaches"""
    
    print("\nBayesian vs Frequentist Comparison")
    print("=" * 50)
    
    # Simulate data
    np.random.seed(42)
    n = 50
    true_p = 0.6
    data = np.random.binomial(1, true_p, n)
    successes = np.sum(data)
    
    # Frequentist
    p_mle = successes / n
    se = np.sqrt(p_mle * (1 - p_mle) / n)
    ci_freq = stats.norm.ppf([0.025, 0.975], p_mle, se)
    
    # Bayesian (Beta(1,1) prior = uniform)
    alpha_post = 1 + successes
    beta_post = 1 + n - successes
    ci_bayes = beta.ppf([0.025, 0.975], alpha_post, beta_post)
    
    print(f"True proportion: {true_p}")
    print(f"Observed: {successes}/{n} = {p_mle:.4f}")
    print(f"\nFrequentist 95% CI: [{ci_freq[0]:.4f}, {ci_freq[1]:.4f}]")
    print(f"Bayesian 95% CI: [{ci_bayes[0]:.4f}, {ci_bayes[1]:.4f}]")
    print(f"\nFrequentist: '95% of such intervals contain true p'")
    print(f"Bayesian: '95% probability p is in this interval'")

bayesian_frequentist_comparison()
```

---

## 📊 Summary Tables

### Conjugate Prior Families

| Likelihood | Conjugate Prior | Posterior |
|------------|-----------------|-----------|
| Bernoulli/Binomial | Beta(α, β) | Beta(α+X, β+n-X) |
| Poisson | Gamma(α, β) | Gamma(α+X, β+1) |
| Normal (known σ²) | Normal(μ₀, σ₀²) | Normal(μₙ, σₙ²) |
| Normal (known μ) | Inverse-Gamma | Inverse-Gamma |

### Bayesian vs Frequentist

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Parameter** | Fixed unknown | Random variable |
| **Probability** | Long-run frequency | Degree of belief |
| **CI Interpretation** | Coverage probability | Credible interval |
| **Prior** | Not used | Incorporated |
| **Computation** | Often analytical | Often MCMC |

---

## 🎯 ML Applications

| Application | Bayesian Concept |
|-------------|-----------------|
| **A/B Testing** | Bayesian inference for conversion rates |
| **Bayesian Optimization** | Posterior over functions |
| **Variational Inference** | Approximate posterior |
| **Bayesian Neural Nets** | Posterior over weights |
| **Uncertainty** | Credible intervals |

---

**Status:** ✅ Complete
**Next:** Update README
