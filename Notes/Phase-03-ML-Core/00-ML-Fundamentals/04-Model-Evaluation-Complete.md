# 6.4 Model Evaluation and Validation

## 🎯 Quick Overview
- **Cross-Validation**: Robust performance estimation
- **Bias-Variance**: Understand model behavior
- **Hyperparameter Tuning**: Optimize performance
- **Foundation for**: Reliable ML models

---

## 1. Cross-Validation

### K-Fold CV

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Stratified CV

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='f1')
```

### Leave-One-Out CV

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

---

## 2. Bias-Variance Tradeoff

### Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0]
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, 'o-', label='Training')
plt.plot(train_sizes, val_mean, 'o-', label='Validation')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### Validation Curves

```python
from sklearn.model_selection import validation_curve

param_range = [1, 10, 50, 100, 200]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(), X, y,
    param_name='n_estimators',
    param_range=param_range, cv=5
)

plt.plot(param_range, train_scores.mean(axis=1), label='Training')
plt.plot(param_range, val_scores.mean(axis=1), label='Validation')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## 3. Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [5, 10, 15, None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

---

## 💻 Python Code Examples

```python
# === Complete Model Evaluation Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cross-Validation
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Test Accuracy: {best_model.score(X_test, y_test):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
```

---

## 📊 Summary Tables

### Cross-Validation Methods

| Method | Folds | Use Case |
|--------|-------|----------|
| K-Fold | k | General purpose |
| Stratified | k | Imbalanced classification |
| LOOCV | n-1 | Very small datasets |

### Hyperparameter Tuning

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Grid Search | Slow | Exhaustive | Small parameter space |
| Random Search | Fast | Good | Large parameter space |
| Bayesian | Medium | Best | Expensive evaluations |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Cross-Validation | Model selection |
| Learning Curves | Debugging models |
| Grid Search | Performance optimization |

---

**Status:** ✅ Complete
**Next:** Feature Engineering
