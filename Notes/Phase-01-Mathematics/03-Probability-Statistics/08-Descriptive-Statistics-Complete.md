# 1.3.8 Descriptive Statistics

## 🎯 Quick Overview
- **Descriptive statistics**: Summarize and describe data
- **Measures of center**: Mean, median, mode
- **Measures of spread**: Variance, standard deviation, IQR
- **Visualization**: Box plots, histograms, Q-Q plots

---

## 1. Population vs Sample

### Population
```
Entire group of interest
Parameters: μ, σ², σ (fixed but often unknown)
```

### Sample
```
Subset of population used for analysis
Statistics: X̄, S², S (calculated from data)
```

### Notation

| Measure | Population | Sample |
|---------|------------|--------|
| Mean | μ | X̄ |
| Variance | σ² | S² |
| Std Dev | σ | S |
| Size | N | n |

---

## 2. Measures of Central Tendency

### Mean (Average)
```
X̄ = (1/n) Σ xᵢ

Pros: Uses all data
Cons: Sensitive to outliers
```

### Median
```
Middle value when sorted

For n values:
- Odd n: middle value
- Even n: average of two middle values

Pros: Robust to outliers
Cons: Doesn't use all information
```

### Mode
```
Most frequent value

Can have multiple modes (bimodal, multimodal)
```

### Comparison

| Measure | Best For | Sensitive to Outliers |
|---------|----------|----------------------|
| Mean | Symmetric distributions | Yes |
| Median | Skewed distributions | No |
| Mode | Categorical data | No |

---

## 3. Measures of Dispersion

### Range
```
Range = max - min

Simple but very sensitive to outliers
```

### Variance
```
Population: σ² = (1/N) Σ(xᵢ - μ)²
Sample: S² = (1/(n-1)) Σ(xᵢ - X̄)²

Note: n-1 for sample (Bessel's correction)
```

### Standard Deviation
```
σ = √σ²
S = √S²

Same units as original data
```

### Interquartile Range (IQR)
```
IQR = Q3 - Q1

Q1 = 25th percentile
Q3 = 75th percentile

Robust to outliers!
```

### Coefficient of Variation
```
CV = σ/μ (or S/X̄)

Expresses variability relative to mean
Unitless - good for comparison
```

---

## 4. Measures of Position

### Percentiles
```
Pₖ = value below which k% of data falls

P₅₀ = median
P₂₅ = Q1
P₇₅ = Q3
```

### Quartiles
```
Q1 = 25th percentile (lower quartile)
Q2 = 50th percentile (median)
Q3 = 75th percentile (upper quartile)
```

### Z-Score (Standard Score)
```
z = (x - μ) / σ  (population)
z = (x - X̄) / S  (sample)

Number of standard deviations from mean
```

---

## 5. Five-Number Summary

```
1. Minimum
2. Q1 (First quartile)
3. Median (Q2)
4. Q3 (Third quartile)
5. Maximum

Provides complete picture of distribution
```

---

## 6. Box Plots (Box-and-Whisker Plots)

### Components
```
┌─────────────────┐
│     whisker     │
│    ┌─────┐      │
│    │ box │      │  Box: Q1 to Q3
│    └─────┘      │  Line in box: median
│     whisker     │  Whiskers: min to max
└─────────────────┘     (or 1.5×IQR)
```

### Outlier Detection
```
Lower fence: Q1 - 1.5×IQR
Upper fence: Q3 + 1.5×IQR

Points outside fences = outliers
```

---

## 7. Histograms and Density Plots

### Histogram
```
Bars show frequency in bins

Bin width affects appearance
Too few bins: oversmoothed
Too many bins: noisy
```

### Density Plot
```
Smoothed version of histogram
Area under curve = 1
Better for comparing distributions
```

---

## 8. Q-Q Plots (Quantile-Quantile)

### Purpose
```
Compare data distribution to theoretical distribution
Usually used to check normality
```

### Interpretation
```
Points on line → matches theoretical distribution
Points curve away → deviation from theoretical

S-curve: heavy tails
Convex: right-skewed
Concave: left-skewed
```

---

## 9. Outlier Detection

### Methods

| Method | Rule | Robust |
|--------|------|--------|
| **Z-score** | |z| > 3 | No |
| **IQR** | Outside Q1±1.5×IQR | Yes |
| **Modified Z-score** | |M| > 3.5 | Yes |

### Modified Z-Score (Robust)
```
M = 0.6745 × (xᵢ - median) / MAD

MAD = median absolute deviation
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 200)

# === Measures of Center ===
print("Measures of Central Tendency")
print("=" * 40)
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Mode: {stats.mode(data, keepdims=True).mode[0]}")

# === Measures of Spread ===
print("\nMeasures of Dispersion")
print("=" * 40)
print(f"Range: {np.max(data) - np.min(data):.2f}")
print(f"Variance: {np.var(data, ddof=1):.2f}")
print(f"Std Dev: {np.std(data, ddof=1):.2f}")

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
print(f"IQR: {IQR:.2f}")
print(f"CV: {np.std(data, ddof=1) / np.mean(data):.4f}")

# === Five-Number Summary ===
print("\nFive-Number Summary")
print("=" * 40)
print(f"Min: {np.min(data):.2f}")
print(f"Q1: {Q1:.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Q3: {Q3:.2f}")
print(f"Max: {np.max(data):.2f}")

# === Visualization ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(data, bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
axes[0, 0].axvline(np.median(data), color='blue', linestyle='--', label=f'Median: {np.median(data):.2f}')
axes[0, 0].set_title('Histogram')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(data, vert=True)
axes[0, 1].set_title('Box Plot')
axes[0, 1].set_ylabel('Value')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].grid(True, alpha=0.3)

# Density plot
x = np.linspace(min(data), max(data), 100)
kde = stats.gaussian_kde(data)
axes[1, 1].plot(x, kde(x), linewidth=2)
axes[1, 1].fill_between(x, kde(x), alpha=0.3)
axes[1, 1].set_title('Density Plot')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Density')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Outlier Detection ===
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower) | (data > upper)]
    return outliers, lower, upper

outliers, lower, upper = detect_outliers_iqr(data)
print(f"\nOutlier Detection (IQR method):")
print(f"Lower fence: {lower:.2f}")
print(f"Upper fence: {upper:.2f}")
print(f"Outliers: {len(outliers)} values")
```

---

## 📊 Summary Tables

### Measures Comparison

| Measure | Formula | Use Case |
|---------|---------|----------|
| Mean | Σxᵢ/n | Symmetric data |
| Median | Middle value | Skewed data |
| Mode | Most frequent | Categorical |
| Variance | Σ(xᵢ-X̄)²/(n-1) | Spread squared |
| Std Dev | √Variance | Same units |
| IQR | Q3-Q1 | Robust spread |

### Visualization Guide

| Plot | Purpose | Best For |
|------|---------|----------|
| Histogram | Distribution shape | Single variable |
| Box Plot | Five-number summary | Comparing groups |
| Q-Q Plot | Normality check | Distribution comparison |
| Density | Smooth distribution | Multiple groups |

---

## 🎯 ML Applications

| Application | Descriptive Statistics |
|-------------|----------------------|
| **Data Exploration** | Summary statistics |
| **Feature Engineering** | Z-scores, normalization |
| **Outlier Detection** | IQR, Z-score methods |
| **Data Quality** | Missing values, ranges |
| **EDA** | Visualizations |

---

**Status:** ✅ Complete
**Next:** Sampling Distributions
