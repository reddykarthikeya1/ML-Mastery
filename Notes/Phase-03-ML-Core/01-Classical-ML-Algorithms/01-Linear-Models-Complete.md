# 7.1 Linear Models

## 🎯 Quick Overview
- **Linear Regression**: Predict continuous values
- **Logistic Regression**: Binary and multi-class classification
- **Regularization**: Ridge, Lasso, Elastic Net
- **GLM**: Generalized Linear Models
- **Foundation for**: Most ML algorithms, statistical modeling

---

#### 🧒 ELI5: Linear Regression, Logistic Regression & Regularization

> Imagine you're trying to predict things based on patterns.
>
> **Linear Regression** (Drawing the best straight line):
>
> You have data: "Hours studied" vs "Test score"
> - Student A: 1 hour → 50 points
> - Student B: 2 hours → 60 points
> - Student C: 3 hours → 70 points
>
> You draw a line: "For every extra hour, +10 points!"
> - Formula: Score = 40 + (10 × Hours)
> - 40 = Base score (even with 0 hours)
> - 10 = How much each hour helps
>
> **Best fit line**: The line that's closest to ALL points
> - Some points above, some below
> - Minimize total distance (that's what OLS does!)
>
> **Logistic Regression** (Yes/No predictions):
>
> Predicting: "Will student pass? (Yes/No)"
> - Can't use straight line (goes to infinity!)
> - Use S-curve (sigmoid): Squishes to 0% - 100%
> - Output: "85% chance of passing"
> - Above 50%? → Predict "Yes, will pass!"
>
> **Why "Regression" if it's classification?**:
> - Still fits a line, just squishes the output!
>
> **Regularization** (Preventing overfitting):
>
> **Problem**: Model memorizes training data too well!
> - "Student who studied 2.347 hours got 63.8 points"
> - Too specific! Won't work on new students
>
> **Ridge (L2)** (Penalize BIG weights):
> - "Having weights over 100 is suspicious!"
> - Shrinks all weights a bit
> - Like: "Don't rely TOO much on any one feature"
>
> **Lasso (L1)** (Eliminate useless features):
> - "Feature with weight 0.001? Just make it ZERO!"
> - Some weights become exactly 0
> - Like: "This feature isn't helping, ignore it!"
> - Does feature selection automatically!
>
> **Elastic Net** (Best of both):
> - Uses BOTH Ridge and Lasso penalties
> - Like: "Shrink weights AND eliminate useless ones"
>
> **Why Regularization works**:
> - Simple models generalize better
> - Complex models memorize
> - Regularization = "Stay simple, stupid!" (Occam's razor)

</details>

---

## 1. Linear Regression

### Ordinary Least Squares (OLS)

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Coefficients
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient: {model.coef_[0][0]:.4f}")
print(f"R² Score: {model.score(X, y):.4f}")
```

### Normal Equation (Closed-Form Solution)

```python
def linear_regression_normal_equation(X, y):
    """Solve linear regression using normal equation"""
    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Normal equation: θ = (X^T X)^(-1) X^T y
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    return theta_best

# Usage
theta_best = linear_regression_normal_equation(X, y)
print(f"Intercept: {theta_best[0][0]:.4f}")
print(f"Slope: {theta_best[1][0]:.4f}")
```

### Gradient Descent for Linear Regression

```python
def linear_regression_gd(X, y, learning_rate=0.1, n_iterations=1000):
    """Solve linear regression using gradient descent"""
    m = len(y)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.random.randn(2, 1)
    
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
    
    return theta

# Usage
theta_gd = linear_regression_gd(X, y)
print(f"GD Intercept: {theta_gd[0][0]:.4f}")
print(f"GD Slope: {theta_gd[1][0]:.4f}")
```

### Assumptions and Diagnostics

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Fit model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Residual analysis
residuals = y - y_pred

# Plot residuals
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Residuals vs Fitted
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# Q-Q plot
from scipy import stats
stats.probplot(residuals.flatten(), dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')

# Residual distribution
sns.histplot(residuals.flatten(), kde=True, ax=axes[2])
axes[2].set_title('Residual Distribution')

plt.tight_layout()
plt.show()

# Check assumptions
print("Assumptions Check:")
print(f"1. Linearity: Check residuals vs fitted plot")
print(f"2. Independence: Check data collection method")
print(f"3. Homoscedasticity: Check constant variance in residuals")
print(f"4. Normality: Check Q-Q plot and histogram")
```

### Multicollinearity and VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def calculate_vif(X):
    """Calculate Variance Inflation Factor"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(len(X.columns))]
    return vif_data

# Example
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)

vif_results = calculate_vif(df)
print(vif_results.sort_values('VIF', ascending=False))

# VIF Interpretation:
# VIF = 1: No multicollinearity
# 1 < VIF < 5: Moderate multicollinearity
# VIF > 5: Severe multicollinearity
```

---

## 2. Regularized Linear Regression

### Ridge Regression (L2 Regularization)

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Cross-validation
scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
print(f"Ridge CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Ridge with different alpha values
alphas = np.logspace(-3, 3, 100)
cv_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
    cv_scores.append(scores.mean())

# Plot
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, cv_scores)
plt.xlabel('Alpha (log scale)')
plt.ylabel('CV R² Score')
plt.title('Ridge Regression: Alpha vs CV Score')
plt.grid(True, alpha=0.3)
plt.show()
```

### Lasso Regression (L1 Regularization)

```python
from sklearn.linear_model import Lasso

# Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Feature selection (coefficients become zero)
print(f"Lasso coefficients: {lasso.coef_}")
print(f"Number of features selected: {np.sum(lasso.coef_ != 0)}")

# Lasso path
from sklearn.linear_model import lasso_path

# Generate data with multiple features
np.random.seed(42)
X_multi = np.random.randn(100, 10)
y_multi = X_multi[:, 0] + 2 * X_multi[:, 1] + np.random.randn(100)

# Lasso path
alphas = np.logspace(-3, 1, 100)
lasso_coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_multi, y_multi)
    lasso_coefs.append(lasso.coef_)

# Plot coefficient paths
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(alphas, [coef[i] for coef in lasso_coefs], label=f'Feature {i}')

plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Paths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Elastic Net

```python
from sklearn.linear_model import ElasticNet

# Elastic Net (L1 + L2 regularization)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_multi, y_multi)

print(f"Elastic Net coefficients: {enet.coef_}")

# Compare Ridge, Lasso, Elastic Net
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

models = {
    'Ridge': Ridge(alpha=0.1),
    'Lasso': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models.items():
    scores = cross_val_score(model, X_multi, y_multi, cv=5, scoring='r2')
    print(f"{name} CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 3. Logistic Regression

### Binary Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

### Sigmoid Function and Decision Boundary

```python
def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))

# Plot sigmoid
z = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid(z), linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision boundary')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Decision boundary visualization
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Train on 2D data
model_2d = LogisticRegression()
model_2d.fit(X_2d, y)

# Plot decision boundary
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='RdBu', edgecolors='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

### Multi-Class Logistic Regression

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load iris dataset (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

# Multi-class logistic regression
model = LogisticRegression(max_iter=1000, multi_class='auto')
model.fit(X, y)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Multi-class CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# One-vs-Rest vs One-vs-One
ovr_model = LogisticRegression(max_iter=1000, multi_class='ovr')
ovo_model = LogisticRegression(max_iter=1000, multi_class='ovo')

ovr_scores = cross_val_score(ovr_model, X, y, cv=5, scoring='accuracy')
ovo_scores = cross_val_score(ovo_model, X, y, cv=5, scoring='accuracy')

print(f"One-vs-Rest Accuracy: {ovr_scores.mean():.4f}")
print(f"One-vs-One Accuracy: {ovo_scores.mean():.4f}")
```

### Class Imbalance Handling

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print(f"Class weights: {class_weights}")

# Train with class weights
model_weighted = LogisticRegression(class_weight='balanced', max_iter=1000)
model_weighted.fit(X_train, y_train)

# Compare with and without class weights
from sklearn.metrics import classification_report

y_pred_unweighted = model.predict(X_test)
y_pred_weighted = model_weighted.predict(X_test)

print("Without class weights:")
print(classification_report(y_test, y_pred_unweighted))

print("\nWith class weights:")
print(classification_report(y_test, y_pred_weighted))
```

---

## 4. Generalized Linear Models (GLM)

### Poisson Regression

```python
import statsmodels.api as sm

# Generate count data
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
true_beta = [0.5, 1.0, -0.5]
mu = np.exp(true_beta[0] + true_beta[1] * X[:, 0] + true_beta[2] * X[:, 1])
y = np.random.poisson(mu)

# Add intercept
X_with_intercept = sm.add_constant(X)

# Poisson regression
model = sm.GLM(y, X_with_intercept, family=sm.families.Poisson())
result = model.fit()

print(result.summary())
```

### Gamma Regression

```python
# Generate positive continuous data
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
true_beta = [0.5, 1.0, -0.5]
mu = np.exp(true_beta[0] + true_beta[1] * X[:, 0] + true_beta[2] * X[:, 1])
y = np.random.gamma(2, mu/2)

# Gamma regression
model = sm.GLM(y, X_with_intercept, family=sm.families.Gamma())
result = model.fit()

print(result.summary())
```

---

## 💻 Python Code Examples

```python
# === Complete Linear Regression Pipeline ===

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load data
boston = load_boston()
X, y = boston.data, boston.target
feature_names = boston.feature_names

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Elastic Net': ElasticNet()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=5, scoring='r2')
    
    results[name] = {
        'RMSE': rmse,
        'R2': r2,
        'CV R2 Mean': cv_scores.mean(),
        'CV R2 Std': cv_scores.std()
    }
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Hyperparameter tuning for Ridge
param_grid = {'alpha': np.logspace(-3, 3, 100)}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest Ridge alpha: {grid_search.best_params_['alpha']}")
print(f"Best Ridge CV R²: {grid_search.best_score_:.4f}")

# === Complete Logistic Regression Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with regularization
model = LogisticRegression(max_iter=1000, C=1.0, penalty='l2')
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
coef_df = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Logistic Regression Feature Importance')
plt.tight_layout()
plt.show()
```

---

## 📊 Summary Tables

### Regularization Comparison

| Method | Penalty | Formula | Use Case |
|--------|---------|---------|----------|
| Ridge | L2 | λΣβ² | Shrink coefficients |
| Lasso | L1 | λΣ|β| | Feature selection |
| Elastic Net | L1 + L2 | λ₁Σ|β| + λ₂Σβ² | Both shrinkage and selection |

### Model Assumptions

| Assumption | Linear Regression | Logistic Regression |
|------------|------------------|---------------------|
| Linearity | ✓ Linear relationship | ✓ Linear log-odds |
| Independence | ✓ Independent errors | ✓ Independent observations |
| Homoscedasticity | ✓ Constant variance | ✓ Not required |
| Normality | ✓ Normal errors | ✓ Not required |
| No Multicollinearity | ✓ Required | ✓ Required |

### GLM Families

| Family | Distribution | Link Function | Use Case |
|--------|-------------|---------------|----------|
| Gaussian | Normal | Identity | Continuous data |
| Poisson | Poisson | Log | Count data |
| Gamma | Gamma | Inverse | Positive continuous |
| Binomial | Binomial | Logit | Binary/proportions |

---

## 🎯 ML Applications

| Algorithm | ML Application |
|-----------|----------------|
| Linear Regression | House price prediction, Sales forecasting |
| Logistic Regression | Spam detection, Disease diagnosis |
| Ridge Regression | Genomics, High-dimensional data |
| Lasso | Feature selection, Compressed sensing |
| Poisson Regression | Count prediction, Risk modeling |

---

## ❓ Quick Check Questions

1. What is the fundamental difference between the Normal Equation and Gradient Descent for solving Linear Regression?
2. Which regularization technique (Lasso or Ridge) is better suited for automatic feature selection, and why?
3. What is the "Sigmoid Function" used for in Logistic Regression?
4. How do you interpret a VIF (Variance Inflation Factor) score of 10?
5. What are the key assumptions of Linear Regression that should be checked using residual plots?

---

## 📝 Answers to Quick Check

1. The **Normal Equation** provides an analytical, closed-form solution in one step ($O(n^3)$ complexity), making it efficient for small datasets. **Gradient Descent** is an iterative optimization algorithm that scales better for very large datasets ($O(kn^2)$ complexity) and works even when the matrix is not invertible.
2. **Lasso (L1 Regularization)** is better for feature selection because its penalty term (absolute value of coefficients) can force some coefficients to become exactly zero, effectively removing those features from the model. Ridge only shrinks them toward zero.
3. The **Sigmoid function** maps any real-valued number into a value between 0 and 1, which can be interpreted as the probability that a given data point belongs to a particular class.
4. A **VIF of 10** indicates severe multicollinearity among the independent variables. Typically, a VIF > 5 or 10 suggests that the associated feature is highly redundant and may need to be removed or combined.
5. The key assumptions are **Linearity** (linear relationship between X and y), **Homoscedasticity** (constant variance of errors), **Independence** of errors, and **Normality** of the error distribution.

---

**Status:** ✅ Complete
**Next:** Tree-Based Models
