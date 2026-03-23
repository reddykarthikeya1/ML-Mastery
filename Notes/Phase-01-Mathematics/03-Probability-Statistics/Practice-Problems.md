# Probability & Statistics - Practice Problems

## Topic 1: Probability Foundations

### Level 1: Basic

**1.1** A fair die is rolled. Find:
- a) P(rolling a 4)
- b) P(rolling even)
- c) P(rolling ≥ 3)

**1.2** Two coins are flipped. What is:
- a) P(both heads)?
- b) P(exactly one head)?
- c) P(at least one head)?

**1.3** In a deck of 52 cards:
- a) P(drawing a King)?
- b) P(drawing a Heart)?
- c) P(drawing King of Hearts)?

---

### Level 2: Intermediate

**1.4** Bayes' Theorem - Medical Testing:
```
Disease prevalence: 1%
Test sensitivity: 99%
Test specificity: 95%

If test is positive, what's P(disease)?
```

**1.5** Urn Problem:
```
Urn A: 3 red, 2 blue
Urn B: 1 red, 4 blue

Choose urn at random, then draw a ball.
If ball is red, what's P(it came from Urn A)?
```

**1.6** Python Practice - Monty Hall Simulation:
```python
import numpy as np

def monty_hall(n_trials=10000, switch=True):
    """
    Simulate Monty Hall problem.
    Return win rate.
    """
    # Your code here
    pass

# Verify that switching wins 2/3 of the time
```

---

## Topic 2: Random Variables and Distributions

### Level 1: Basic

**2.1** X ~ Bernoulli(p = 0.7). Find:
- a) E[X]
- b) Var(X)
- c) P(X = 1)

**2.2** X ~ Binomial(n = 10, p = 0.5). Find:
- a) E[X]
- b) Var(X)
- c) P(X = 5)

**2.3** X ~ N(μ = 100, σ = 15). Find:
- a) P(X > 115)
- b) P(85 < X < 115)
- c) 90th percentile

---

### Level 2: Intermediate

**2.4** Waiting Time:
```
Calls arrive at Poisson rate λ = 5 per hour.

a) P(no calls in 1 hour)?
b) P(at least 3 calls in 1 hour)?
c) Expected time until first call?
```

**2.5** Normal Approximation:
```
Coin flipped 100 times.

a) P(exactly 50 heads)?
b) P(more than 60 heads)?
Use normal approximation.
```

**2.6** Python Practice - Distribution Visualization:
```python
from scipy import stats
import matplotlib.pyplot as plt

# Plot PMF/PDF of:
# 1. Binomial(n=20, p=0.3)
# 2. Poisson(λ=5)
# 3. Normal(μ=0, σ=1)
# 4. Exponential(λ=1)
```

---

## Topic 3: Joint Distributions

### Level 2: Intermediate

**3.1** Joint PMF:
```
     Y=0   Y=1   Y=2
X=0  0.1   0.2   0.1
X=1  0.2   0.3   0.1

Find:
a) Marginal P(X=1)
b) Conditional P(Y=1 | X=0)
c) Cov(X, Y)
d) Are X and Y independent?
```

**3.2** Bivariate Normal:
```
(X, Y) ~ BVN(μₓ=0, μᵧ=0, σₓ²=1, σᵧ²=1, ρ=0.5)

Find:
a) E[X | Y = 1]
b) Var(X | Y = 1)
```

---

## Topic 4: Limit Theorems

### Level 2: Intermediate

**4.1** Chebyshev's Inequality:
```
Distribution with μ = 50, σ = 10.

Find lower bound for P(30 < X < 70).
```

**4.2** Central Limit Theorem:
```
Population: Exponential(λ = 1)
Sample size: n = 100

a) Approximate distribution of sample mean?
b) P(sample mean > 1.1)?
```

**4.3** Python Practice - CLT Demonstration:
```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate CLT with uniform distribution
# Show sample means approach normal as n increases

def clt_demo():
    # Your code here
    pass
```

---

## Topic 5: Estimation and Hypothesis Testing

### Level 2: Intermediate

**5.1** MLE:
```
Data: x₁, ..., xₙ from Exponential(λ)

Find MLE of λ.
```

**5.2** Confidence Interval:
```
Sample: n = 100, X̄ = 50, s = 10

Find 95% CI for population mean.
```

**5.3** Hypothesis Test:
```
H₀: μ = 100
H₁: μ ≠ 100

Sample: n = 25, X̄ = 105, s = 15

a) Calculate t-statistic
b) Find p-value
c) Decision at α = 0.05
```

---

## Topic 6: Regression

### Level 2: Intermediate

**6.1** Simple Linear Regression:
```
Data:
X: 1, 2, 3, 4, 5
Y: 2, 4, 5, 4, 5

Find:
a) Least squares estimates β̂₀, β̂₁
b) R²
c) Prediction for X = 6
```

**6.2** Python Practice - Regression Analysis:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate data: Y = 3 + 2X + ε
np.random.seed(42)
X = np.random.uniform(0, 10, 100).reshape(-1, 1)
Y = 3 + 2 * X + np.random.normal(0, 1, 100).reshape(-1, 1)

# Fit linear regression
# Report coefficients and R²
# Plot residuals
```

---

## Topic 7: Information Theory

### Level 2: Intermediate

**7.1** Entropy:
```
a) Entropy of fair coin?
b) Entropy of biased coin (p = 0.9)?
c) Entropy of fair die?
```

**7.2** KL Divergence:
```
p = [0.5, 0.5]
q = [0.8, 0.2]

Calculate D_KL(p || q) and D_KL(q || p).
```

**7.3** Python Practice - Information Measures:
```python
import numpy as np

def entropy(p):
    """Calculate entropy of distribution p"""
    # Your code here
    pass

def kl_divergence(p, q):
    """Calculate KL divergence"""
    # Your code here
    pass

# Test with various distributions
```

---

## Solutions (Selected Problems)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
```
a) 1/6
b) 3/6 = 1/2
c) 4/6 = 2/3
```

### 1.4
```
P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|¬D)P(¬D)]
       = 0.99×0.01 / [0.99×0.01 + 0.05×0.99]
       = 0.0099 / 0.0594
       = 0.167 ≈ 16.7%
```

### 2.1
```
a) E[X] = p = 0.7
b) Var(X) = p(1-p) = 0.21
c) P(X=1) = p = 0.7
```

### 2.3
```
a) P(X > 115) = P(Z > 1) = 0.1587
b) P(85 < X < 115) = P(-1 < Z < 1) = 0.6827
c) 90th percentile: μ + 1.28σ = 100 + 1.28×15 = 119.2
```

### 4.1
```
P(30 < X < 70) = P(|X - 50| < 20)
               = P(|X - μ| < 2σ)
≥ 1 - 1/2² = 1 - 1/4 = 0.75

At least 75% of values are between 30 and 70.
```

### 5.2
```
95% CI: X̄ ± 1.96 × s/√n
      = 50 ± 1.96 × 10/10
      = 50 ± 1.96
      = [48.04, 51.96]
```

### 6.1
```
a) β̂₁ = 0.7, β̂₀ = 2.5
b) R² = 0.7
c) Ŷ = 2.5 + 0.7×6 = 6.7
```

### 7.1
```
a) H = 1 bit
b) H = 0.469 bits
c) H = log₂(6) = 2.585 bits
```

</details>

---

## 📝 Notes Section

Use this space for additional problems:

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Probability & Statistics Complete!
