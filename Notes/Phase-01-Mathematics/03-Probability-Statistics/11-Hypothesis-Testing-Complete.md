# 1.3.11 Hypothesis Testing

## 🎯 Quick Overview
- **Hypothesis testing**: Make decisions based on data
- **p-values**: Evidence against null hypothesis
- **Common tests**: z-tests, t-tests, chi-squared, ANOVA
- **Foundation for**: A/B testing, model comparison, scientific inference

---

## 1. Null and Alternative Hypotheses

### Null Hypothesis (H₀)
```
Status quo, no effect, no difference
What we test against
Assumed true until evidence suggests otherwise
```

### Alternative Hypothesis (H₁ or Hₐ)
```
Research hypothesis, effect exists
What we want to prove
```

### Examples

| Scenario | H₀ | H₁ |
|----------|-----|-----|
| Drug effectiveness | No effect | Drug works |
| A/B test | A = B | A ≠ B |
| Quality control | μ = target | μ ≠ target |

---

## 2. Types of Errors

| Decision | H₀ True | H₀ False |
|----------|---------|----------|
| **Don't reject H₀** | ✓ Correct (1-α) | Type II error (β) |
| **Reject H₀** | Type I error (α) | ✓ Correct (Power=1-β) |

### Significance Level (α)
```
Probability of Type I error
Common values: 0.05, 0.01, 0.001

P(reject H₀ | H₀ true) = α
```

### Power (1-β)
```
Probability of correctly rejecting H₀
P(reject H₀ | H₀ false) = 1-β

Typical target: 80% power
```

---

## 3. p-Values

### Definition
```
p-value = P(observing data as extreme as ours | H₀ true)

Small p-value → Strong evidence against H₀
```

### Interpretation

| p-value | Evidence against H₀ |
|---------|---------------------|
| < 0.001 | Very strong |
| 0.001-0.01 | Strong |
| 0.01-0.05 | Moderate |
| 0.05-0.10 | Weak |
| > 0.10 | Little to none |

### Decision Rule
```
If p-value < α: Reject H₀
If p-value ≥ α: Fail to reject H₀

Note: "Fail to reject" ≠ "Accept H₀"
```

---

## 4. One-Tailed vs Two-Tailed Tests

### Two-Tailed Test
```
H₀: μ = μ₀
H₁: μ ≠ μ₀

Tests for difference in either direction
```

### One-Tailed Test
```
H₀: μ ≤ μ₀ (or μ ≥ μ₀)
H₁: μ > μ₀ (or μ < μ₀)

Tests for difference in one direction
More powerful but must justify direction
```

---

## 5. z-Tests

### One-Sample z-Test
```
When: σ known, large sample

Test statistic:
z = (X̄ - μ₀) / (σ/√n)

Critical values (α=0.05, two-tailed): ±1.96
```

### Two-Sample z-Test
```
Compare two means, σ₁ and σ₂ known

z = (X̄₁ - X̄₂) / √(σ₁²/n₁ + σ₂²/n₂)
```

---

## 6. t-Tests

### One-Sample t-Test
```
H₀: μ = μ₀

t = (X̄ - μ₀) / (s/√n)

df = n-1
```

### Independent Two-Sample t-Test
```
H₀: μ₁ = μ₂

t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)

df ≈ n₁ + n₂ - 2
```

### Paired t-Test
```
H₀: μ_d = 0 (mean difference = 0)

t = d̄ / (s_d/√n)

df = n-1

Use for: Before/after, matched pairs
```

### Welch's t-Test
```
When variances are unequal

t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)

df calculated using Welch-Satterthwaite equation
```

---

## 7. Chi-Squared Tests

### Goodness of Fit Test
```
H₀: Data follows specified distribution

χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ

df = k - 1 - (number of estimated parameters)
```

### Test of Independence
```
H₀: Two categorical variables are independent

χ² = Σ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ

df = (r-1)(c-1)

Contingency table:
          Col1  Col2  ...
Row1      O₁₁   O₁₂   ...
Row2      O₂₁   O₂₂   ...
...
```

---

## 8. F-Tests

### Comparing Two Variances
```
H₀: σ₁² = σ₂²

F = S₁² / S₂²  (larger variance on top)

df₁ = n₁ - 1, df₂ = n₂ - 1
```

### ANOVA (Analysis of Variance)
```
H₀: μ₁ = μ₂ = ... = μₖ

F = MS_between / MS_within

df_between = k - 1
df_within = N - k
```

---

## 9. ANOVA

### One-Way ANOVA
```
Compare means across k groups

F = MS_between / MS_within

If F is significant → at least one group differs
```

### Two-Way ANOVA
```
Two factors (independent variables)

Tests:
- Main effect of Factor A
- Main effect of Factor B
- Interaction effect A×B
```

### Post-hoc Tests
```
After significant ANOVA, find which groups differ:

- Tukey's HSD (honestly significant difference)
- Bonferroni correction
- Scheffé's method
```

---

## 10. Non-Parametric Tests

### When to Use
```
- Data not normally distributed
- Small sample sizes
- Ordinal data
- Outliers present
```

### Common Non-Parametric Tests

| Parametric | Non-Parametric | Use Case |
|------------|----------------|----------|
| One-sample t | Sign test, Wilcoxon signed-rank | Median test |
| Two-sample t | Mann-Whitney U | Independent samples |
| Paired t | Wilcoxon signed-rank | Paired samples |
| ANOVA | Kruskal-Wallis | k independent groups |

---

## 11. Multiple Testing Correction

### Problem
```
With many tests, false positives accumulate

If α = 0.05 and 100 tests:
Expected false positives = 5
```

### Bonferroni Correction
```
α_corrected = α / m

where m = number of tests

Conservative but simple
```

### False Discovery Rate (FDR)
```
Control proportion of false positives among rejections

Benjamini-Hochberg procedure:
1. Order p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)
2. Find largest k where p(k) ≤ (k/m)·α
3. Reject H₀ for all tests 1 to k
```

---

## 💻 Python Code Examples

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# === One-Sample t-Test ===

print("One-Sample t-Test")
print("=" * 50)

# Sample data
sample = np.array([23, 25, 28, 22, 26, 24, 27, 25, 23, 26, 24, 25])
mu_0 = 24

# Perform t-test
t_stat, p_value = stats.ttest_1samp(sample, mu_0)

print(f"Sample mean: {np.mean(sample):.2f}")
print(f"Test value (μ₀): {mu_0}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# === Two-Sample t-Test ===

print("\nIndependent Two-Sample t-Test")
print("=" * 50)

np.random.seed(42)
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(105, 15, 50)

t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"Group 1 mean: {np.mean(group1):.2f}")
print(f"Group 2 mean: {np.mean(group2):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# === Paired t-Test ===

print("\nPaired t-Test")
print("=" * 50)

before = np.array([85, 90, 78, 92, 88, 76, 95, 89, 82, 91])
after = np.array([88, 92, 80, 95, 90, 79, 97, 91, 85, 94])

t_stat, p_value = stats.ttest_rel(after, before)

print(f"Before mean: {np.mean(before):.2f}")
print(f"After mean: {np.mean(after):.2f}")
print(f"Mean difference: {np.mean(after - before):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# === Chi-Squared Test ===

print("\nChi-Squared Test of Independence")
print("=" * 50)

# Contingency table
observed = np.array([[45, 55], [60, 40]])  # Rows: Treatment, Control; Cols: Success, Failure

chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"Observed frequencies:")
print(observed)
print(f"\nExpected frequencies:")
print(expected)
print(f"\nχ² statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# === ANOVA ===

print("\nOne-Way ANOVA")
print("=" * 50)

np.random.seed(42)
group_a = np.random.normal(100, 15, 30)
group_b = np.random.normal(105, 15, 30)
group_c = np.random.normal(95, 15, 30)

f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)

print(f"Group A mean: {np.mean(group_a):.2f}")
print(f"Group B mean: {np.mean(group_b):.2f}")
print(f"Group C mean: {np.mean(group_c):.2f}")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# === Mann-Whitney U Test (Non-parametric) ===

print("\nMann-Whitney U Test")
print("=" * 50)

# Non-normal data
group1 = np.random.exponential(5, 50)
group2 = np.random.exponential(7, 50)

u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

print(f"Group 1 median: {np.median(group1):.2f}")
print(f"Group 2 median: {np.median(group2):.2f}")
print(f"U statistic: {u_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# === Multiple Testing Correction ===

def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction"""
    m = len(p_values)
    alpha_corrected = alpha / m
    rejections = [p < alpha_corrected for p in p_values]
    return alpha_corrected, rejections

def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction"""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Find largest k where p(k) ≤ (k/m)·α
    thresholds = (np.arange(1, m+1) / m) * alpha
    significant = sorted_p <= thresholds
    k = np.max(np.where(significant)[0]) + 1 if np.any(significant) else 0
    
    rejections = np.zeros(m, dtype=bool)
    rejections[sorted_indices[:k]] = True
    
    return k/m * alpha, rejections

# Example
np.random.seed(42)
n_tests = 100
p_values = np.random.uniform(0, 1, n_tests)  # Simulated p-values

bonf_alpha, bonf_rej = bonferroni_correction(p_values)
fdr_alpha, fdr_rej = benjamini_hochberg(p_values)

print("\nMultiple Testing Correction")
print("=" * 50)
print(f"Number of tests: {n_tests}")
print(f"\nBonferroni:")
print(f"  Corrected α: {bonf_alpha:.6f}")
print(f"  Rejections: {sum(bonf_rej)}")
print(f"\nBenjamini-Hochberg (FDR):")
print(f"  Rejections: {sum(fdr_rej)}")

# === Power Analysis ===

from statsmodels.stats.power import TTestIndPower

print("\nPower Analysis")
print("=" * 50)

# Effect size (Cohen's d)
effect_size = 0.5  # Medium effect
alpha = 0.05
power = 0.80

analysis = TTestIndPower()
sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)

print(f"Effect size (Cohen's d): {effect_size}")
print(f"Desired power: {power}")
print(f"Significance level: {alpha}")
print(f"\nRequired sample size per group: {int(np.ceil(sample_size))}")
```

---

## 📊 Summary Tables

### Common Tests

| Test | Use Case | Test Statistic |
|------|----------|----------------|
| **z-test** | Known σ, large n | z = (X̄-μ₀)/(σ/√n) |
| **t-test** | Unknown σ, small n | t = (X̄-μ₀)/(s/√n) |
| **Chi-squared** | Categorical data | χ² = Σ(O-E)²/E |
| **F-test** | Compare variances | F = S₁²/S₂² |
| **ANOVA** | Compare k means | F = MS_between/MS_within |

### Error Types

| Error Type | Symbol | Meaning |
|------------|--------|---------|
| Type I | α | False positive |
| Type II | β | False negative |
| Power | 1-β | Correct detection |

---

## 🎯 ML Applications

| Application | Hypothesis Testing |
|-------------|-------------------|
| **A/B Testing** | Compare conversion rates |
| **Feature Selection** | Test feature importance |
| **Model Comparison** | Compare model performance |
| **Quality Assurance** | Monitor model drift |

---

**Status:** ✅ Complete
**Next:** Regression Analysis
