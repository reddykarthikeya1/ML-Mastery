# 6.4 Model Evaluation and Validation

## 🎯 Quick Overview
- **Cross-Validation**: Robust performance estimation
- **Bias-Variance**: Understand model behavior
- **Hyperparameter Tuning**: Optimize model performance
- **Foundation for**: Reliable ML models

---

## 1. Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Basic K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Scores: {scores}")
```

### Stratified Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

# For classification with imbalanced classes
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skfold, scoring='f1')

print(f"Stratified CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Leave-One-Out CV

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

print(f"LOOCV Accuracy: {scores.mean():.4f}")
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# For time-dependent data
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold score: {score:.4f}")
```

### Custom Cross-Validation

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

# Multiple metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric in scoring:
    print(f"{metric}: {cv_results[f'test_{metric}'].mean():.4f}")
```

---

## 2. Bias-Variance Tradeoff

### Understanding Bias and Variance

```
Total Error = Bias² + Variance + Irreducible Error

Bias: Error from erroneous assumptions
- High bias → Underfitting
- Model too simple
- Solution: More features, less regularization

Variance: Error from sensitivity to training data
- High variance → Overfitting
- Model too complex
- Solution: More data, regularization, simpler model
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Generate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
    n_jobs=-1
)

# Calculate statistics
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training')
plt.fill_between(train_sizes, train_mean - train_std, 
                train_mean + train_std, alpha=0.1)

plt.plot(train_sizes, val_mean, 'o-', label='Validation')
plt.fill_between(train_sizes, val_mean - val_std, 
                val_mean + val_std, alpha=0.1)

plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Validation Curves

```python
from sklearn.model_selection import validation_curve

# Vary one hyperparameter
param_range = [1, 10, 50, 100, 200, 500]

train_scores, val_scores = validation_curve(
    RandomForestClassifier(), X, y,
    param_name='n_estimators',
    param_range=param_range,
    cv=5, scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'o-', label='Training')
plt.plot(param_range, val_mean, 'o-', label='Validation')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.legend()
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3. Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit
grid_search.fit(X_train, y_train)

# Results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test Score: {test_score:.4f}")

# All results
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values('rank_test_score')
print(results[['params', 'mean_test_score', 'rank_test_score']].head())
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Parameter distributions
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.5, 0.5)
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.4f}")
```

### Bayesian Optimization

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Parameter space
search_spaces = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.5, 1.0)
}

# BayesSearchCV
bayes_search = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    search_spaces,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)

print(f"Best Parameters: {bayes_search.best_params_}")
print(f"Best CV Score: {bayes_search.best_score_:.4f}")
```

---

## 4. Model Selection

### Comparing Multiple Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Plot comparison
plt.figure(figsize=(12, 6))
names = list(results.keys())
means = [results[name]['mean'] for name in names]
stds = [results[name]['std'] for name in names]

plt.bar(names, means, yerr=stds, capsize=5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.tight_layout()
plt.show()
```

### Statistical Tests

```python
from scipy import stats

# Paired t-test
model1_scores = cross_val_score(model1, X, y, cv=5)
model2_scores = cross_val_score(model2, X, y, cv=5)

t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference between models")
else:
    print("No significant difference")
```

---

## 5. Model Diagnostics

### Residual Analysis

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Residual plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residual Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
```

### Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

# SHAP values (for detailed interpretation)
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

---

## 💻 Python Code Examples

```python
# === Complete Model Evaluation Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cross-validation
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nTest Accuracy: {best_model.score(X_test, y_test):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # Top 10

plt.figure(figsize=(12, 6))
plt.title('Top 10 Feature Importances')
plt.bar(range(10), importances[indices])
plt.xticks(range(10), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

# === Learning Curve Analysis ===

from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy',
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_learning_curve(RandomForestClassifier(), X, y)
```

---

## 📊 Summary Tables

### Cross-Validation Methods

| Method | Folds | Use Case | Pros | Cons |
|--------|-------|----------|------|------|
| K-Fold | k | General | Balanced | May miss stratification |
| Stratified | k | Imbalanced | Preserves distribution | Slightly slower |
| LOOCV | n-1 | Very small | Maximum training | Very slow |
| Time Series | k | Time-dependent | Respects time order | Less data per fold |

### Hyperparameter Tuning

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Grid Search | Slow | Exhaustive | Small parameter space |
| Random Search | Fast | Good | Large parameter space |
| Bayesian | Medium | Best | Expensive evaluations |

### Bias vs Variance

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Low train, low val accuracy | High bias | More features, less regularization |
| High train, low val accuracy | High variance | More data, regularization |
| Both high | Good fit | - |

---

## 🎯 ML Applications

| Evaluation Technique | ML Application |
|---------------------|----------------|
| Cross-Validation | Model selection |
| Learning Curves | Debugging models |
| Hyperparameter Tuning | Performance optimization |
| Feature Importance | Model interpretation |

---

---

## ❓ Quick Check Questions

1. Why is Stratified K-Fold cross-validation preferred over standard K-Fold for classification tasks?
2. What are the two primary components of Total Error in an ML model?
3. How can you identify a "High Variance" (Overfitting) problem using a Learning Curve?
4. What is the main computational disadvantage of Grid Search compared to Random Search?
5. When should you use Time Series cross-validation instead of standard K-Fold?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Stratified K-Fold** ensures that each fold maintains the same percentage of samples for each class as the original dataset. This is crucial for imbalanced datasets where a standard random split might result in folds with zero samples of the minority class.
2. Total Error = **Bias² + Variance + Irreducible Error**.
3. **High Variance** is indicated by a large gap between the training accuracy and the validation accuracy. The training accuracy is usually very high, while the validation accuracy significantly lags behind.
4. **Grid Search** is exhaustive and scales exponentially with the number of hyperparameters and their possible values (the "Curse of Dimensionality"). **Random Search** is much faster as it only evaluates a fixed number of random combinations, often finding a near-optimal solution in far less time.
5. Use **Time Series cross-validation** when the data has a temporal dependency (e.g., stock prices, weather). Standard K-Fold would "leak" information from the future into the past, violating the chronological order of the data.

</details>

---

**Status:** ✅ Complete
**Next:** Feature Engineering
