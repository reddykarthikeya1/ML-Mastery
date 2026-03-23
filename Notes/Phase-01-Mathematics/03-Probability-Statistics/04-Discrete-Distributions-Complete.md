# 1.3.4 Important Discrete Distributions

## 🎯 Quick Overview
- **Bernoulli**: Single yes/no trial
- **Binomial**: Count successes in n trials
- **Poisson**: Events per time period
- **Foundation for**: Classification, count data, rare events

---

## 1. Bernoulli Distribution

### Definition

**Single trial with two outcomes:**
```
X ~ Bernoulli(p)

PMF: P(X = x) = pˣ(1-p)¹⁻ˣ  for x ∈ {0, 1}

Or explicitly:
P(X = 1) = p  (success)
P(X = 0) = 1-p  (failure)
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| p | Probability of success | 0 ≤ p ≤ 1 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = p |
| **Variance** | Var(X) = p(1-p) |
| **Skewness** | (1-2p)/√(p(1-p)) |

### Examples

```
Coin flip: X = 1 if heads, 0 if tails
p = 0.5 for fair coin

Yes/No survey: X = 1 if yes, 0 if no
p = proportion who say yes

Defective item: X = 1 if defective, 0 if good
p = defect rate
```

### Python

```python
from scipy import stats

p = 0.7
bernoulli = stats.bernoulli(p)

print(f"P(X=1) = {bernoulli.pmf(1)}")  # 0.7
print(f"Mean = {bernoulli.mean()}")    # 0.7
print(f"Variance = {bernoulli.var()}") # 0.21
```

---

## 2. Binomial Distribution

### Definition

**Number of successes in n independent Bernoulli trials:**
```
X ~ Binomial(n, p)

PMF: P(X = k) = C(n,k) · pᵏ · (1-p)ⁿ⁻ᵏ

for k = 0, 1, 2, ..., n
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| n | Number of trials | n ≥ 1, integer |
| p | Success probability | 0 ≤ p ≤ 1 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = np |
| **Variance** | Var(X) = np(1-p) |
| **Skewness** | (1-2p)/√(np(1-p)) |

### Why Binomial?

```
C(n,k) = number of ways to choose k successes from n trials
pᵏ = probability of k successes
(1-p)ⁿ⁻ᵏ = probability of n-k failures

Multiply: ways × probability of each way
```

### Examples

**Example 1: Coin Flips**
```
Flip 10 coins, count heads:
X ~ Binomial(n=10, p=0.5)

P(X = 5) = C(10,5) · 0.5⁵ · 0.5⁵
         = 252 · 0.03125 · 0.03125
         = 0.246

E[X] = 10 · 0.5 = 5
Var(X) = 10 · 0.5 · 0.5 = 2.5
```

**Example 2: Quality Control**
```
Defect rate = 5%, sample 20 items:
X = number of defective items

X ~ Binomial(n=20, p=0.05)

P(X = 0) = C(20,0) · 0.05⁰ · 0.95²⁰
         = 1 · 1 · 0.358
         = 0.358

P(X ≥ 1) = 1 - P(X = 0) = 1 - 0.358 = 0.642

E[X] = 20 · 0.05 = 1
Var(X) = 20 · 0.05 · 0.95 = 0.95
```

### Relationship to Bernoulli

```
X ~ Binomial(n, p) can be written as:

X = X₁ + X₂ + ... + Xₙ  where Xᵢ ~ Bernoulli(p)

Sum of n independent Bernoulli trials
```

### Python

```python
from scipy import stats

n, p = 10, 0.5
binom = stats.binom(n, p)

# PMF
print(f"P(X=5) = {binom.pmf(5):.4f}")

# CDF
print(f"P(X≤3) = {binom.cdf(3):.4f}")

# Mean and variance
print(f"Mean = {binom.mean()}")
print(f"Variance = {binom.var()}")

# Visualization
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, n+1)
pmf = binom.pmf(x)

plt.bar(x, pmf)
plt.title(f'Binomial(n={n}, p={p})')
plt.xlabel('k')
plt.ylabel('P(X=k)')
plt.show()
```

---

## 3. Geometric Distribution

### Definition

**Number of trials until first success:**
```
X ~ Geometric(p)

PMF: P(X = k) = (1-p)ᵏ⁻¹ · p

for k = 1, 2, 3, ...
```

### Alternative Definition

**Number of failures before first success:**
```
P(X = k) = (1-p)ᵏ · p  for k = 0, 1, 2, ...
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| p | Success probability | 0 < p ≤ 1 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = 1/p |
| **Variance** | Var(X) = (1-p)/p² |
| **Skewness** | (2-p)/√(1-p) |

### Memoryless Property

```
P(X > m + n | X > m) = P(X > n)

"Past failures don't affect future probability"

Example: If coin landed tails 10 times, 
P(heads on next flip) is still 0.5
```

### Examples

**Example 3: First Success**
```
p = 0.2 (20% success rate)

P(X = 1) = 0.8⁰ · 0.2 = 0.2  (success on first try)
P(X = 2) = 0.8¹ · 0.2 = 0.16 (success on second try)
P(X = 3) = 0.8² · 0.2 = 0.128

P(X ≤ 3) = 0.2 + 0.16 + 0.128 = 0.488

E[X] = 1/0.2 = 5 trials on average
```

### Python

```python
from scipy import stats

p = 0.2
geom = stats.geom(p)

print(f"P(X=1) = {geom.pmf(1):.4f}")
print(f"P(X≤5) = {geom.cdf(5):.4f}")
print(f"Mean = {geom.mean():.4f}")
print(f"Variance = {geom.var():.4f}")
```

---

## 4. Poisson Distribution

### Definition

**Number of events in fixed time/space:**
```
X ~ Poisson(λ)

PMF: P(X = k) = e^(-λ) · λᵏ / k!

for k = 0, 1, 2, ...
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| λ | Rate (events per interval) | λ > 0 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = λ |
| **Variance** | Var(X) = λ |
| **Skewness** | 1/√λ |

**Note:** Mean = Variance for Poisson!

### When to Use Poisson

**Poisson processes:**
- Events occur independently
- Constant average rate λ
- Events don't occur simultaneously

**Examples:**
- Emails per hour
- Customers per day
- Defects per meter
- Radioactive decays per second

### Relationship to Binomial

```
If n is large and p is small:

Binomial(n, p) ≈ Poisson(λ = np)

Rule of thumb: n ≥ 20 and p ≤ 0.05
```

### Examples

**Example 4: Call Center**
```
Average 5 calls per hour: λ = 5

P(X = 3) = e^(-5) · 5³ / 3!
         = 0.00674 · 125 / 6
         = 0.140

P(X = 0) = e^(-5) = 0.0067

P(X ≥ 1) = 1 - P(X = 0) = 0.9933

E[X] = 5
Var(X) = 5
```

**Example 5: Rare Events**
```
Defect rate 0.1%, sample 1000 items:
X = number of defects

Exact: X ~ Binomial(1000, 0.001)
Approx: X ~ Poisson(λ = 1000 × 0.001 = 1)

P(X = 0) ≈ e^(-1) = 0.368
P(X ≥ 2) = 1 - P(X=0) - P(X=1)
         = 1 - 0.368 - 0.368
         = 0.264
```

### Python

```python
from scipy import stats

λ = 5
poisson = stats.poisson(λ)

# PMF
x = np.arange(0, 15)
pmf = poisson.pmf(x)

# CDF
print(f"P(X≤3) = {poisson.cdf(3):.4f}")
print(f"P(X>10) = {1 - poisson.cdf(10):.4f}")

# Mean and variance
print(f"Mean = {poisson.mean()}")
print(f"Variance = {poisson.var()}")

# Plot
plt.bar(x, pmf)
plt.title(f'Poisson(λ={λ})')
plt.xlabel('k')
plt.ylabel('P(X=k)')
plt.show()
```

---

## 5. Negative Binomial Distribution

### Definition

**Number of failures before r successes:**
```
X ~ NegativeBinomial(r, p)

PMF: P(X = k) = C(k+r-1, r-1) · pʳ · (1-p)ᵏ

for k = 0, 1, 2, ...
```

### Parameters

| Parameter | Meaning | Range |
|-----------|---------|-------|
| r | Number of successes | r ≥ 1 |
| p | Success probability | 0 < p ≤ 1 |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = r(1-p)/p |
| **Variance** | Var(X) = r(1-p)/p² |

### Relationship to Geometric

```
NegativeBinomial(r, p) = Sum of r independent Geometric(p)

When r = 1: NegativeBinomial = Geometric
```

### Examples

```
Keep flipping until 3 heads (r=3), p=0.5

X = number of tails before 3rd head

E[X] = 3(0.5)/0.5 = 3
Var(X) = 3(0.5)/0.25 = 6
```

---

## 6. Hypergeometric Distribution

### Definition

**Sampling without replacement:**
```
X ~ Hypergeometric(N, K, n)

Population: N items, K successes
Sample: n items (without replacement)
X = number of successes in sample

PMF: P(X = k) = C(K,k) · C(N-K, n-k) / C(N, n)

for max(0, n+K-N) ≤ k ≤ min(n, K)
```

### Parameters

| Parameter | Meaning |
|-----------|---------|
| N | Population size |
| K | Successes in population |
| n | Sample size |

### Moments

| Moment | Value |
|--------|-------|
| **Mean** | E[X] = n·K/N |
| **Variance** | Var(X) = n·(K/N)·(1-K/N)·(N-n)/(N-1) |

### vs Binomial

| Hypergeometric | Binomial |
|----------------|----------|
| Without replacement | With replacement |
| Dependent trials | Independent trials |
| Variance has correction factor | Simple variance |

### Examples

```
Deck of 52 cards, K=13 hearts
Draw n=5 cards without replacement

X = number of hearts in hand

P(X = 2) = C(13,2) · C(39,3) / C(52,5)
         = 78 · 9139 / 2598960
         = 0.274

E[X] = 5 · 13/52 = 1.25
```

---

## 7. Multinomial Distribution

### Definition

**Generalization of binomial to k categories:**
```
(X₁, X₂, ..., Xₖ) ~ Multinomial(n, p₁, p₂, ..., pₖ)

PMF: P(X₁=x₁, ..., Xₖ=xₖ) = n!/(x₁!...xₖ!) · p₁^x₁ · ... · pₖ^xₖ

where Σxᵢ = n and Σpᵢ = 1
```

### Moments

| Moment | Value |
|--------|-------|
| **E[Xᵢ]** | n·pᵢ |
| **Var(Xᵢ)** | n·pᵢ·(1-pᵢ) |
| **Cov(Xᵢ, Xⱼ)** | -n·pᵢ·pⱼ (negative: more of one means less of another) |

### Examples

```
Roll die 10 times, count each face:
(X₁, X₂, X₃, X₄, X₅, X₆) ~ Multinomial(10, 1/6, ..., 1/6)

E[Xᵢ] = 10 · 1/6 = 5/3
Var(Xᵢ) = 10 · 1/6 · 5/6 = 50/36
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Compare Discrete Distributions ===

def compare_distributions():
    """Compare different discrete distributions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Bernoulli
    ax = axes[0, 0]
    p = 0.7
    bernoulli = stats.bernoulli(p)
    x = [0, 1]
    pmf = bernoulli.pmf(x)
    ax.bar(x, pmf, alpha=0.7)
    ax.set_title(f'Bernoulli(p={p})')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X=x)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Failure', 'Success'])
    
    # 2. Binomial
    ax = axes[0, 1]
    n, p = 20, 0.3
    binom = stats.binom(n, p)
    x = np.arange(0, n+1)
    pmf = binom.pmf(x)
    ax.bar(x, pmf, alpha=0.7)
    ax.axvline(binom.mean(), color='r', linestyle='--', label=f'Mean = {binom.mean()}')
    ax.set_title(f'Binomial(n={n}, p={p})')
    ax.set_xlabel('k')
    ax.set_ylabel('P(X=k)')
    ax.legend()
    
    # 3. Poisson
    ax = axes[1, 0]
    λ = 5
    poisson = stats.poisson(λ)
    x = np.arange(0, 15)
    pmf = poisson.pmf(x)
    ax.bar(x, pmf, alpha=0.7)
    ax.axvline(poisson.mean(), color='r', linestyle='--', label=f'Mean = {poisson.mean()}')
    ax.set_title(f'Poisson(λ={λ})')
    ax.set_xlabel('k')
    ax.set_ylabel('P(X=k)')
    ax.legend()
    
    # 4. Geometric
    ax = axes[1, 1]
    p = 0.3
    geom = stats.geom(p)
    x = np.arange(1, 15)
    pmf = geom.pmf(x)
    ax.bar(x, pmf, alpha=0.7)
    ax.axvline(geom.mean(), color='r', linestyle='--', label=f'Mean = {geom.mean():.2f}')
    ax.set_title(f'Geometric(p={p})')
    ax.set_xlabel('k')
    ax.set_ylabel('P(X=k)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

compare_distributions()

# === Binomial Convergence to Normal ===

def binomial_to_normal():
    """Show binomial converging to normal as n increases"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    p = 0.5
    n_values = [5, 10, 30, 100]
    
    for ax, n in zip(axes.flatten(), n_values):
        binom = stats.binom(n, p)
        x = np.arange(0, n+1)
        pmf = binom.pmf(x)
        
        ax.bar(x, pmf, alpha=0.5, label='Binomial')
        
        # Normal approximation
        mu, sigma = n*p, np.sqrt(n*p*(1-p))
        x_norm = np.linspace(0, n, 200)
        norm_pdf = stats.norm.pdf(x_norm, mu, sigma)
        ax.plot(x_norm, norm_pdf, 'r-', linewidth=2, label='Normal approx')
        
        ax.set_title(f'Binomial(n={n}, p={p})')
        ax.set_xlabel('k')
        ax.set_ylabel('P(X=k)')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

binomial_to_normal()

# === Poisson Approximation to Binomial ===

def poisson_approximation():
    """Demonstrate Poisson approximation to Binomial"""
    
    n = 100
    p = 0.02
    λ = n * p
    
    binom = stats.binom(n, p)
    poisson = stats.poisson(λ)
    
    x = np.arange(0, 15)
    
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - 0.2, binom.pmf(x), width=0.4, alpha=0.7, label=f'Binomial(n={n}, p={p})')
    plt.bar(x + 0.2, poisson.pmf(x), width=0.4, alpha=0.7, label=f'Poisson(λ={λ})')
    
    plt.title(f'Poisson Approximation to Binomial\n(λ = np = {λ})')
    plt.xlabel('k')
    plt.ylabel('P(X=k)')
    plt.legend()
    plt.show()
    
    # Compare probabilities
    print(f"{'k':<5} {'Binomial':<15} {'Poisson':<15} {'Difference'}")
    print("-" * 50)
    for k in range(0, 8):
        binom_p = binom.pmf(k)
        pois_p = poisson.pmf(k)
        print(f"{k:<5} {binom_p:<15.6f} {pois_p:<15.6f} {abs(binom_p - pois_p):.6f}")

poisson_approximation()

# === Expected Values Verification ===

def verify_expectations():
    """Verify theoretical vs sample expectations"""
    
    print("=" * 60)
    print("THEORETICAL VS SAMPLE MOMENTS")
    print("=" * 60)
    
    distributions = {
        'Bernoulli(0.7)': (stats.bernoulli(0.7), 0.7, 0.7*0.3),
        'Binomial(20, 0.3)': (stats.binom(20, 0.3), 6, 4.2),
        'Poisson(5)': (stats.poisson(5), 5, 5),
        'Geometric(0.3)': (stats.geom(0.3), 1/0.3, 0.7/0.09),
    }
    
    n_samples = 100000
    
    print(f"{'Distribution':<20} {'E[X] (th)':<12} {'E[X] (sm)':<12} {'Var(th)':<12} {'Var(sm)':<12}")
    print("-" * 70)
    
    for name, (dist, E_th, Var_th) in distributions.items():
        samples = dist.rvs(n_samples)
        E_sm = np.mean(samples)
        Var_sm = np.var(samples)
        
        print(f"{name:<20} {E_th:<12.4f} {E_sm:<12.4f} {Var_th:<12.4f} {Var_sm:<12.4f}")

verify_expectations()

# === Real-world Applications ===

def real_world_examples():
    """Real-world applications of discrete distributions"""
    
    print("\n" + "=" * 60)
    print("REAL-WORLD APPLICATIONS")
    print("=" * 60)
    
    # Example 1: A/B Testing
    print("\n1. A/B Testing (Binomial)")
    print("   Scenario: Website conversion rates")
    print("   Version A: 100 conversions from 1000 visitors (p=0.10)")
    print("   Version B: 130 conversions from 1000 visitors")
    print("   Question: Is B significantly better than A?")
    
    # Example 2: Call Center
    print("\n2. Call Center Staffing (Poisson)")
    λ = 50  # calls per hour
    poisson = stats.poisson(λ)
    
    # Find how many staff needed
    for capacity in [40, 50, 55, 60, 65]:
        prob = 1 - poisson.cdf(capacity)
        print(f"   P(more than {capacity} calls) = {prob:.4f}")
    
    # Example 3: Quality Control
    print("\n3. Quality Control (Hypergeometric)")
    print("   Lot of 1000 items, 50 defective")
    print("   Sample 50 items, accept if ≤ 2 defective")
    
    hypergeom = stats.hypergeom(1000, 50, 50)
    p_accept = hypergeom.cdf(2)
    print(f"   P(accept lot) = {p_accept:.4f}")

real_world_examples()
```

---

## 📊 Summary Table

| Distribution | PMF | Mean | Variance | Use Case |
|--------------|-----|------|----------|----------|
| **Bernoulli(p)** | pˣ(1-p)¹⁻ˣ | p | p(1-p) | Single trial |
| **Binomial(n,p)** | C(n,k)pᵏ(1-p)ⁿ⁻ᵏ | np | np(1-p) | Count successes |
| **Geometric(p)** | (1-p)ᵏ⁻¹p | 1/p | (1-p)/p² | Trials to success |
| **Poisson(λ)** | e^(-λ)λᵏ/k! | λ | λ | Events per time |
| **NegBin(r,p)** | C(k+r-1,r-1)pʳ(1-p)ᵏ | r(1-p)/p | r(1-p)/p² | Failures to r successes |
| **Hypergeom(N,K,n)** | C(K,k)C(N-K,n-k)/C(N,n) | nK/N | n(K/N)(1-K/N)(N-n)/(N-1) | Sampling w/o replacement |

---

## 🎯 ML Applications

| Application | Distribution |
|-------------|--------------|
| **Binary Classification** | Bernoulli (output) |
| **Count Prediction** | Poisson regression |
| **A/B Testing** | Binomial proportions |
| **Rare Event Detection** | Poisson |
| **Multi-class Classification** | Multinomial |

---

**Status:** ✅ Complete
**Next:** Continuous Distributions
