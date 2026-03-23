# 1.3.12 Regression Analysis

## 🎯 Quick Overview
- **Regression**: Model relationship between variables
- **Least squares**: Minimize prediction errors
- **R²**: Proportion of variance explained
- **Foundation for**: Prediction, causal inference, ML models

---

## 1. Simple Linear Regression

### Model

```
Y = β₀ + β₁X + ε

where:
- Y = Dependent variable (outcome)
- X = Independent variable (predictor)
- β₀ = Intercept
- β₁ = Slope
- ε = Error term ~ N(0, σ²)
```

### Assumptions

1. **Linearity**: Relationship is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant error variance
4. **Normality**: Errors are normally distributed

---

## 2. Least Squares Estimation

### Objective

```
Minimize sum of squared residuals:

SSE = Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - (β₀ + β₁xᵢ))²
```

### Estimates

```
β̂₁ = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
   = Cov(X,Y) / Var(X)
   = r · (sᵧ/sₓ)

β̂₀ = ȳ - β̂₁x̄
```

### Fitted Values and Residuals

```
ŷᵢ = β̂₀ + β̂₁xᵢ  (predicted value)

eᵢ = yᵢ - ŷᵢ  (residual)

Properties:
- Σeᵢ = 0
- Σxᵢeᵢ = 0
- Σŷᵢeᵢ = 0
```

---

## 3. Model Evaluation

### Coefficient of Determination (R²)

```
R² = 1 - SS_res/SS_tot
   = SS_reg/SS_tot
   = (Correlation)²

where:
- SS_tot = Σ(yᵢ - ȳ)²  (total sum of squares)
- SS_reg = Σ(ŷᵢ - ȳ)²  (regression sum of squares)
- SS_res = Σ(yᵢ - ŷᵢ)²  (residual sum of squares)

Interpretation: Proportion of variance in Y explained by X
0 ≤ R² ≤ 1
```

### Standard Error of Estimate

```
s = √(SS_res / (n-2))

RMSE = √(SS_res / n)

Measures typical prediction error
```

---

## 4. Inference for Regression

### Hypothesis Test for Slope

```
H₀: β₁ = 0  (no relationship)
H₁: β₁ ≠ 0  (relationship exists)

Test statistic:
t = β̂₁ / SE(β̂₁)

df = n - 2
```

### Confidence Interval for Slope

```
β̂₁ ± t* · SE(β̂₁)

where t* from t-distribution with df = n-2
```

### Confidence vs Prediction Intervals

```
Confidence interval for mean response at X = x₀:
ŷ ± t* · s · √(1/n + (x₀-x̄)²/Σ(xᵢ-x̄)²)

Prediction interval for individual response:
ŷ ± t* · s · √(1 + 1/n + (x₀-x̄)²/Σ(xᵢ-x̄)²)

Prediction intervals are wider!
```

---

## 5. Correlation vs Causation

### Correlation

```
Measures strength of linear relationship

r = Σ(xᵢ-x̄)(yᵢ-ȳ) / √(Σ(xᵢ-x̄)² · Σ(yᵢ-ȳ)²)

-1 ≤ r ≤ 1
```

### Causation

```
Correlation does NOT imply causation!

Possible explanations for correlation:
1. X causes Y
2. Y causes X (reverse causation)
3. Z causes both X and Y (confounding)
4. Coincidence
```

---

## 6. Residual Analysis

### Check Assumptions

| Assumption | Diagnostic Plot |
|------------|-----------------|
| Linearity | Residuals vs Fitted |
| Constant variance | Residuals vs Fitted |
| Normality | Q-Q plot of residuals |
| Independence | Residuals vs Order |

### Patterns to Look For

```
Good: Random scatter around 0

Problems:
- Funnel shape → Heteroscedasticity
- Curve pattern → Non-linearity
- Outliers → Points far from rest
- Trends → Non-independence
```

---

## 7. Multiple Linear Regression

### Model

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε

Matrix form: Y = Xβ + ε
```

### Least Squares Solution

```
β̂ = (X'X)⁻¹ X'Y

Ŷ = Xβ̂

e = Y - Ŷ
```

### Assumptions (Extended)

1. No perfect multicollinearity
2. Correct specification
3. Exogeneity: E[ε|X] = 0

---

## 8. Model Selection

### Adjusted R²

```
R²_adj = 1 - (1-R²)(n-1)/(n-k-1)

Penalizes for number of predictors
Can decrease when adding useless predictors
```

### AIC and BIC

```
AIC = n·ln(SSE/n) + 2k
BIC = n·ln(SSE/n) + k·ln(n)

Lower is better
BIC penalizes complexity more heavily
```

---

## 9. Polynomial Regression

### Model

```
Y = β₀ + β₁X + β₂X² + ... + βₖXᵏ + ε

Still linear in parameters!
```

### Use When

```
Curved relationship
Diminishing/increasing returns
U-shaped or inverted U-shaped patterns
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import statsmodels.api as sm

# === Simple Linear Regression ===

np.random.seed(42)

# Generate data
n = 100
X = np.random.uniform(0, 10, n).reshape(-1, 1)
true_beta0, true_beta1 = 5, 2
Y = true_beta0 + true_beta1 * X + np.random.normal(0, 2, n).reshape(-1, 1)

# Fit model
model = LinearRegression()
model.fit(X, Y)

y_pred = model.predict(X)

# Calculate statistics
SS_tot = np.sum((Y - np.mean(Y))**2)
SS_res = np.sum((Y - y_pred)**2)
R_squared = 1 - SS_res / SS_tot

print("Simple Linear Regression")
print("=" * 50)
print(f"True: Y = {true_beta0} + {true_beta1}X + ε")
print(f"Fitted: Y = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f}X")
print(f"R² = {R_squared:.4f}")
print(f"RMSE = {np.sqrt(SS_res/n):.4f}")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, Y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Fitted line')
plt.title(f'Simple Linear Regression (R² = {R_squared:.3f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(1, 2, 2)
residuals = Y - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Using statsmodels for Inference ===

print("\nRegression with Inference")
print("=" * 50)

# Add constant for intercept
X_sm = sm.add_constant(X)
model_sm = sm.OLS(Y, X_sm).fit()

print(model_sm.summary())

# === Multiple Regression ===

print("\nMultiple Linear Regression")
print("=" * 50)

# Generate multiple predictors
n = 200
X1 = np.random.uniform(0, 10, n)
X2 = np.random.uniform(0, 5, n)
X3 = np.random.uniform(0, 3, n)

# True model: Y = 10 + 2*X1 - 1.5*X2 + 0.5*X3 + ε
Y = 10 + 2*X1 - 1.5*X2 + 0.5*X3 + np.random.normal(0, 1, n)

# Fit model
X_multi = np.column_stack([X1, X2, X3])
X_multi_sm = sm.add_constant(X_multi)
model_multi = sm.OLS(Y, X_multi_sm).fit()

print(model_multi.summary())

# === Polynomial Regression ===

print("\nPolynomial Regression")
print("=" * 50)

# Generate curved data
n = 100
X_poly = np.random.uniform(-3, 3, n).reshape(-1, 1)
Y_poly = 5 - 2*X_poly + 0.5*X_poly**2 + np.random.normal(0, 0.5, n).reshape(-1, 1)

# Fit polynomial regression
degree = 2
poly = PolynomialFeatures(degree=degree)
X_poly_transformed = poly.fit_transform(X_poly)

model_poly = LinearRegression()
model_poly.fit(X_poly_transformed, Y_poly)

y_poly_pred = model_poly.predict(X_poly_transformed)

# Calculate R²
SS_tot = np.sum((Y_poly - np.mean(Y_poly))**2)
SS_res = np.sum((Y_poly - y_poly_pred)**2)
R_squared_poly = 1 - SS_res / SS_tot

print(f"Polynomial degree: {degree}")
print(f"Coefficients: {model_poly.coef_}")
print(f"R² = {R_squared_poly:.4f}")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_poly, Y_poly, alpha=0.5, label='Data')
X_sorted = np.sort(X_poly, axis=0)
X_sorted_transformed = poly.transform(X_sorted)
y_sorted_pred = model_poly.predict(X_sorted_transformed)
plt.plot(X_sorted, y_sorted_pred, 'r-', linewidth=2, label='Fitted curve')
plt.title(f'Polynomial Regression (degree={degree}, R²={R_squared_poly:.3f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(1, 2, 2)
residuals_poly = Y_poly - y_poly_pred
plt.scatter(y_poly_pred, residuals_poly, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Polynomial Regression - Residuals')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Correlation vs Causation Demo ===

def correlation_causation_demo():
    """Demonstrate that correlation does not imply causation"""
    
    np.random.seed(42)
    n = 100
    
    # Confounding variable
    Z = np.random.normal(0, 1, n)  # e.g., socioeconomic status
    
    # Both X and Y are caused by Z
    X = 2*Z + np.random.normal(0, 0.5, n)  # e.g., education level
    Y = 3*Z + np.random.normal(0, 0.5, n)  # e.g., income
    
    # X and Y are correlated but don't cause each other
    correlation = np.corrcoef(X, Y)[0, 1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5)
    plt.title(f'X and Y are correlated (r={correlation:.3f})\nBut both are caused by confounding variable Z')
    plt.xlabel('X (e.g., Education)')
    plt.ylabel('Y (e.g., Income)')
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    plt.annotate('Spurious correlation!\n(Z causes both)', 
                xy=(0.5, 0.5), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=12)
    
    plt.show()
    
    print(f"Correlation between X and Y: {correlation:.4f}")
    print("But X does NOT cause Y - both are caused by Z!")

correlation_causation_demo()

# === Confidence and Prediction Intervals ===

def plot_intervals():
    """Plot confidence and prediction intervals"""
    
    np.random.seed(42)
    n = 50
    X = np.random.uniform(0, 10, n).reshape(-1, 1)
    Y = 5 + 2*X + np.random.normal(0, 2, n).reshape(-1, 1)
    
    # Fit model
    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)
    
    # Calculate intervals
    X_new = np.linspace(0, 10, 100).reshape(-1, 1)
    y_new_pred = model.predict(X_new)
    
    # Standard error
    MSE = np.sum((Y - y_pred)**2) / (n - 2)
    X_bar = np.mean(X)
    SS_xx = np.sum((X - X_bar)**2)
    
    # Confidence interval for mean response
    SE_mean = np.sqrt(MSE * (1/n + (X_new - X_bar)**2 / SS_xx))
    t_crit = stats.t.ppf(0.975, n-2)
    ci_lower = y_new_pred - t_crit * SE_mean
    ci_upper = y_new_pred + t_crit * SE_mean
    
    # Prediction interval for individual response
    SE_pred = np.sqrt(MSE * (1 + 1/n + (X_new - X_bar)**2 / SS_xx))
    pi_lower = y_new_pred - t_crit * SE_pred
    pi_upper = y_new_pred + t_crit * SE_pred
    
    plt.figure(figsize=(12, 8))
    plt.scatter(X, Y, alpha=0.5, label='Data')
    plt.plot(X_new, y_new_pred, 'r-', linewidth=2, label='Fitted line')
    
    plt.fill_between(X_new.flatten(), ci_lower.flatten(), ci_upper.flatten(), 
                    alpha=0.3, color='blue', label='95% Confidence Interval (mean)')
    plt.fill_between(X_new.flatten(), pi_lower.flatten(), pi_upper.flatten(), 
                    alpha=0.3, color='green', label='95% Prediction Interval (individual)')
    
    plt.title('Confidence vs Prediction Intervals')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Blue band: Confidence interval for mean response")
    print("Green band: Prediction interval for individual response")
    print("Prediction intervals are wider because they include individual variation!")

plot_intervals()
```

---

## 📊 Summary Tables

### Regression Formulas

| Concept | Formula |
|---------|---------|
| Slope | β̂₁ = Cov(X,Y)/Var(X) |
| Intercept | β̂₀ = ȳ - β̂₁x̄ |
| R² | 1 - SS_res/SS_tot |
| RMSE | √(SS_res/n) |
| t-test for slope | t = β̂₁/SE(β̂₁) |

### Assumptions Check

| Assumption | Diagnostic | Fix if Violated |
|------------|------------|-----------------|
| Linearity | Residuals vs Fitted | Transform variables |
| Constant variance | Residuals vs Fitted | Weighted regression |
| Normality | Q-Q plot | Transform Y |
| Independence | Residuals vs Order | Time series methods |

---

## 🎯 ML Applications

| Application | Regression Concept |
|-------------|-------------------|
| **Linear Regression** | Baseline model |
| **Polynomial Features** | Non-linear patterns |
| **Regularization** | Ridge, Lasso (prevent overfitting) |
| **Feature Importance** | Coefficient magnitude |
| **Uncertainty** | Prediction intervals |

---

**Status:** ✅ Complete
**Next:** Bayesian Statistics
