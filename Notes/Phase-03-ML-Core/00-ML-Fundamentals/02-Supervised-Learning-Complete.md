# 6.2 Supervised Learning

## 🎯 Quick Overview
- **Supervised Learning**: Learn from labeled data
- **Regression**: Predict continuous values
- **Classification**: Predict categories
- **Foundation for**: Most ML applications

---

## 1. Regression

### Simple Linear Regression

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

# Predict
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Coefficients
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient: {model.coef_[0][0]:.4f}")
print(f"R² Score: {model.score(X, y):.4f}")
```

### Multiple Linear Regression

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
boston = load_boston()
X, y = boston.data, boston.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

### Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Polynomial regression
poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2)),
    ('linear_regression', LinearRegression())
])

poly_model.fit(X, y)
y_poly_pred = poly_model.predict(X)
```

---

## 2. Classification

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```

### Multi-Class Classification

```python
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier

# Load iris dataset (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

# One-vs-Rest
ovr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
ovr_model.fit(X, y)

# Predict
y_pred = ovr_model.predict(X)
```

---

## 3. Metrics

### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MAE
mae = mean_absolute_error(y_test, y_pred)

# MSE
mse = mean_squared_error(y_test, y_pred)

# RMSE
rmse = np.sqrt(mse)

# R²
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# TN FP
# FN TP

# Derived metrics
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# ROC-AUC
from sklearn.metrics import roc_curve
y_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)

print(f"AUC-ROC: {auc_score:.4f}")
```

---

## 💻 Python Code Examples

```python
# === Complete Supervised Learning Pipeline ===

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Regression Example ===

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                                n_informative=8, noise=10, random_state=42)

# Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg_scaled, y_train_reg)

# Predict
y_pred_reg = rf_reg.predict(X_test_reg_scaled)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Regression RMSE: {rmse:.4f}")
print(f"Regression R²: {r2:.4f}")

# === Classification Example ===

# Generate classification data
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, 
                                    n_informative=15, n_redundant=5,
                                    random_state=42)

# Split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Scale
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_clf_scaled, y_train_clf)

# Predict
y_pred_clf = rf_clf.predict(X_test_clf_scaled)

# Evaluate
print(f"\nClassification Report:\n{classification_report(y_test_clf, y_pred_clf)}")

# Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-Validation
cv_scores = cross_val_score(rf_clf, X_train_clf_scaled, y_train_clf, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 📊 Summary Tables

### Regression Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | avg(\|y-ŷ\|) | Average absolute error |
| MSE | avg((y-ŷ)²) | Squared error |
| RMSE | √MSE | Same units as y |
| R² | 1 - SS_res/SS_tot | Variance explained |

### Classification Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| Precision | TP/(TP+FP) | Minimize false positives |
| Recall | TP/(TP+FN) | Minimize false negatives |
| F1 Score | 2×Prec×Rec/(Prec+Rec) | Balance precision/recall |
| AUC-ROC | Area under ROC curve | Overall performance |

### Algorithm Selection

| Problem | Algorithm | When to Use |
|---------|-----------|-------------|
| Linear relationship | Linear Regression | Simple, interpretable |
| Binary classification | Logistic Regression | Probability output |
| Multi-class | Random Forest | High accuracy |
| Non-linear | Random Forest | Complex patterns |

---

## 🎯 ML Applications

| Task | Algorithm | Industry |
|------|-----------|----------|
| House Pricing | Linear Regression | Real Estate |
| Spam Detection | Logistic Regression | Tech |
| Disease Diagnosis | Random Forest | Healthcare |
| Customer Churn | Random Forest | Telecom |

---

---

## ❓ Quick Check Questions

1. What is the main difference between Linear Regression and Logistic Regression?
2. What does the R² score represent?
3. What is the purpose of the Sigmoid function in Logistic Regression?
4. Define Precision and Recall.
5. What does a Confusion Matrix show?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Linear Regression** is used for regression tasks (predicting continuous numerical values). **Logistic Regression** is used for classification tasks (predicting discrete categories/probabilities).
2. **R² (Coefficient of Determination)** represents the proportion of variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1.
3. The **Sigmoid function** maps any real-valued number into a value between 0 and 1, which can be interpreted as the probability of a class.
4. **Precision:** The ratio of correctly predicted positive observations to the total predicted positives. **Recall:** The ratio of correctly predicted positive observations to all actual positives.
5. A **Confusion Matrix** is a table used to evaluate the performance of a classification model, showing the counts of True Positives, True Negatives, False Positives, and False Negatives.

</details>

---

**Status:** ✅ Complete
**Next:** Unsupervised Learning
