# 1.3.8-10 Estimation, Hypothesis Testing, and Regression

## 🎯 Quick Overview
- **Estimation**: Infer population parameters from samples
- **Hypothesis Testing**: Make decisions based on data
- **Regression**: Model relationships between variables
- **Foundation for**: All statistical inference, ML model evaluation

---

## Part 1: Estimation Theory

### 1. Point Estimation

**Estimator:** Rule/function for estimating parameter

**Estimate:** Specific value from sample

**Common Estimators:**
```
Sample mean:     X̄ = (1/n) ΣXᵢ  → estimates μ
Sample variance: S² = (1/(n-1)) Σ(Xᵢ - X̄)² → estimates σ²
Sample proportion: p̂ = X/n → estimates p
```

### 2. Properties of Estimators

| Property | Definition |
|----------|------------|
| **Unbiasedness** | E[θ̂] = θ |
| **Consistency** | θ̂ → θ as n → ∞ |
| **Efficiency** | Minimum variance among unbiased estimators |
| **Sufficiency** | Contains all information about parameter |

### 3. Maximum Likelihood Estimation (MLE)

**Likelihood:**
```
L(θ | x₁,...,xₙ) = f(x₁,...,xₙ | θ) = Π f(xᵢ | θ)

Log-likelihood: ℓ(θ) = log L(θ)
```

**MLE:**
```
θ̂_MLE = argmax L(θ) = argmax ℓ(θ)
```

**Examples:**
```
Normal mean:     μ̂ = X̄
Normal variance: σ̂² = (1/n) Σ(Xᵢ - X̄)²
Bernoulli:       p̂ = X̄ (sample proportion)
Poisson:         λ̂ = X̄
```

### 4. Confidence Intervals

**Definition:** Range that likely contains parameter

**For mean (known σ):**
```
X̄ ± z* · σ/√n

where z* depends on confidence level:
90%: z* = 1.645
95%: z* = 1.96
99%: z* = 2.576
```

**For mean (unknown σ):**
```
X̄ ± t* · s/√n

where t* from t-distribution with n-1 df
```

**Interpretation:**
```
"95% confidence" means:
If we repeated this many times, 95% of intervals would contain μ

NOT: "95% probability that μ is in this interval"
```

---

## Part 2: Hypothesis Testing

### 1. Basic Framework

**Null Hypothesis (H₀):** Status quo, no effect

**Alternative Hypothesis (H₁):** What we're testing for

**Test Statistic:** Function of data

**p-value:** Probability of observed data (or more extreme) if H₀ is true

### 2. Decision Rule

```
If p-value < α: Reject H₀ (statistically significant)
If p-value ≥ α: Fail to reject H₀

Common α levels: 0.05, 0.01, 0.001
```

### 3. Types of Errors

| | H₀ True | H₀ False |
|-|-----------|-------------|
| **Don't reject H₀** | ✓ Correct | Type II error (β) |
| **Reject H₀** | Type I error (α) | ✓ Correct (Power = 1-β) |

### 4. Common Tests

**One-sample z-test:**
```
H₀: μ = μ₀

z = (X̄ - μ₀) / (σ/√n)
```

**One-sample t-test:**
```
H₀: μ = μ₀

t = (X̄ - μ₀) / (s/√n)

df = n - 1
```

**Two-sample t-test:**
```
H₀: μ₁ = μ₂

t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

**Chi-squared test:**
```
χ² = Σ (O - E)² / E

df = (rows - 1)(cols - 1)
```

---

## Part 3: Regression Analysis

### 1. Simple Linear Regression

**Model:**
```
Y = β₀ + β₁X + ε

where ε ~ N(0, σ²)
```

**Least Squares Estimates:**
```
β̂₁ = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)² = Cov(X,Y) / Var(X)
β̂₀ = ȳ - β̂₁x̄
```

**Predictions:**
```
Ŷ = β̂₀ + β̂₁X
```

### 2. Model Evaluation

**R-squared:**
```
R² = 1 - SS_res/SS_tot
   = (Correlation)²

Proportion of variance explained
0 ≤ R² ≤ 1
```

**Assumptions:**
1. Linearity
2. Independence
3. Homoscedasticity (constant variance)
4. Normality of residuals

### 3. Multiple Regression

**Model:**
```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε

Matrix form: Y = Xβ + ε
```

**Solution:**
```
β̂ = (X'X)⁻¹ X'Y
```

**Adjusted R²:**
```
R²_adj = 1 - (1-R²)(n-1)/(n-k-1)

Penalizes for number of predictors
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# === MLE Example ===

def mle_example():
    """Maximum Likelihood Estimation example"""
    
    np.random.seed(42)
    
    # Generate data from N(μ=5, σ=2)
    true_mu, true_sigma = 5, 2
    data = np.random.normal(true_mu, true_sigma, 1000)
    
    # MLE estimates
    mu_mle = np.mean(data)
    sigma_mle = np.std(data)  # MLE uses n, not n-1
    
    print("MLE for Normal Distribution")
    print("=" * 50)
    print(f"True μ = {true_mu}, MLE μ̂ = {mu_mle:.4f}")
    print(f"True σ = {true_sigma}, MLE σ̂ = {sigma_mle:.4f}")
    
    # Visualize
    x = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, label='Data')
    plt.plot(x, stats.norm.pdf(x, true_mu, true_sigma), 'r-', linewidth=2, label='True')
    plt.plot(x, stats.norm.pdf(x, mu_mle, sigma_mle), 'b--', linewidth=2, label='MLE')
    plt.title('MLE for Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

mle_example()

# === Confidence Intervals ===

def confidence_intervals():
    """Demonstrate confidence intervals"""
    
    np.random.seed(42)
    
    true_mean = 100
    true_std = 15
    n = 50
    n_simulations = 100
    
    confidence_levels = [0.90, 0.95, 0.99]
    
    print("\nConfidence Interval Simulation")
    print("=" * 60)
    
    for conf in confidence_levels:
        z = stats.norm.ppf(1 - (1-conf)/2)
        contains = 0
        
        for _ in range(n_simulations):
            sample = np.random.normal(true_mean, true_std, n)
            sample_mean = np.mean(sample)
            se = true_std / np.sqrt(n)
            
            ci_lower = sample_mean - z * se
            ci_upper = sample_mean + z * se
            
            if ci_lower <= true_mean <= ci_upper:
                contains += 1
        
        coverage = contains / n_simulations
        print(f"{conf*100:.0f}% CI: {coverage:.2%} coverage (expected: {conf:.2%})")

confidence_intervals()

# === Hypothesis Testing ===

def hypothesis_testing():
    """Demonstrate hypothesis testing"""
    
    np.random.seed(42)
    
    print("\nHypothesis Testing Examples")
    print("=" * 60)
    
    # One-sample t-test
    sample = np.random.normal(102, 15, 50)
    t_stat, p_value = stats.ttest_1samp(sample, 100)
    
    print("\nOne-sample t-test (H₀: μ = 100):")
    print(f"Sample mean: {np.mean(sample):.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
    
    # Two-sample t-test
    group1 = np.random.normal(100, 15, 50)
    group2 = np.random.normal(105, 15, 50)
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    print("\nTwo-sample t-test (H₀: μ₁ = μ₂):")
    print(f"Group 1 mean: {np.mean(group1):.2f}")
    print(f"Group 2 mean: {np.mean(group2):.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
    
    # Chi-squared test
    observed = np.array([45, 55, 50, 50])
    expected = np.array([50, 50, 50, 50])
    chi2, p_value = stats.chisquare(observed, expected)
    
    print("\nChi-squared test:")
    print(f"Observed: {observed}")
    print(f"Expected: {expected}")
    print(f"χ² statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")

hypothesis_testing()

# === Linear Regression ===

def linear_regression_demo():
    """Demonstrate linear regression"""
    
    np.random.seed(42)
    
    # Generate data
    n = 100
    X = np.random.uniform(0, 10, n).reshape(-1, 1)
    true_beta0, true_beta1 = 5, 2
    Y = true_beta0 + true_beta1 * X + np.random.normal(0, 3, n)
    
    # Fit model
    model = LinearRegression()
    model.fit(X, Y)
    
    y_pred = model.predict(X)
    
    # Calculate R²
    ss_res = np.sum((Y - y_pred)²)
    ss_tot = np.sum((Y - np.mean(Y))²)
    r_squared = 1 - ss_res / ss_tot
    
    print("\nLinear Regression")
    print("=" * 60)
    print(f"True: Y = {true_beta0} + {true_beta1}X + ε")
    print(f"Fitted: Y = {model.intercept_:.2f} + {model.coef_[0]:.2f}X")
    print(f"R² = {r_squared:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5, label='Data')
    plt.plot(X, y_pred, 'r-', linewidth=2, label='Fitted line')
    plt.title(f'Linear Regression (R² = {r_squared:.3f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Residual analysis
    residuals = Y - y_pred
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()

linear_regression_demo()
```

---

## 📊 Summary Tables

### Estimation

| Concept | Formula | Use |
|---------|---------|-----|
| **Sample Mean** | X̄ = (1/n)ΣXᵢ | Estimate μ |
| **Sample Variance** | S² = (1/(n-1))Σ(Xᵢ-X̄)² | Estimate σ² |
| **MLE** | θ̂ = argmax L(θ) | General estimation |
| **CI for μ** | X̄ ± z*·σ/√n | Interval estimate |

### Hypothesis Testing

| Test | Statistic | Use |
|------|-----------|-----|
| **z-test** | z = (X̄-μ₀)/(σ/√n) | Known σ |
| **t-test** | t = (X̄-μ₀)/(s/√n) | Unknown σ |
| **Chi-squared** | χ² = Σ(O-E)²/E | Categorical data |

### Regression

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Slope** | β̂₁ = Cov(X,Y)/Var(X) | Change in Y per unit X |
| **Intercept** | β̂₀ = ȳ - β̂₁x̄ | Y when X = 0 |
| **R²** | 1 - SS_res/SS_tot | Variance explained |

---

## 🎯 ML Applications

| Application | Statistics Concept |
|-------------|-------------------|
| **Model Training** | MLE (maximize likelihood) |
| **Model Evaluation** | Hypothesis testing, R² |
| **A/B Testing** | t-tests, confidence intervals |
| **Feature Selection** | Regression coefficients |
| **Uncertainty** | Confidence intervals |

---

**Status:** ✅ Complete
**Next:** Information Theory
