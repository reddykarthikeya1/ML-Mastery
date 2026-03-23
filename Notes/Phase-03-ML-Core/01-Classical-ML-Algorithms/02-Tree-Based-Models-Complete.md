# 7.2 Tree-Based Models

## 🎯 Quick Overview
- **Decision Trees**: Interpretable, non-linear
- **Random Forest**: Ensemble of trees
- **Boosting**: Sequential improvement
- **XGBoost/LightGBM/CatBoost**: Modern implementations
- **Foundation for**: Most accurate tabular ML

---

## 1. Decision Trees

### Basic Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

# Evaluate
accuracy = tree.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
importances = tree.feature_importances_
for feature, importance in zip(iris.feature_names, importances):
    print(f"{feature}: {importance:.4f}")
```

### Tree Visualization

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.show()
```

### Splitting Criteria

```python
from sklearn.tree import DecisionTreeClassifier

# Gini impurity
tree_gini = DecisionTreeClassifier(criterion='gini')
tree_gini.fit(X_train, y_train)

# Entropy (Information Gain)
tree_entropy = DecisionTreeClassifier(criterion='entropy')
tree_entropy.fit(X_train, y_train)

# Compare
print(f"Gini Accuracy: {tree_gini.score(X_test, y_test):.4f}")
print(f"Entropy Accuracy: {tree_entropy.score(X_test, y_test):.4f}")
```

---

## 2. Ensemble Methods - Bagging

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
accuracy = rf.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xlabel('Feature Rank')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

### Extra Trees

```python
from sklearn.ensemble import ExtraTreesClassifier

# Extremely Randomized Trees
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et.fit(X_train, y_train)

print(f"Extra Trees Accuracy: {et.score(X_test, y_test):.4f}")
```

---

## 3. Ensemble Methods - Boosting

### AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost
ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {ada.score(X_test, y_test):.4f}")
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)
print(f"Gradient Boosting Accuracy: {gb.score(X_test, y_test):.4f}")
```

### XGBoost

```python
import xgboost as xgb

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
print(f"XGBoost Accuracy: {xgb_model.score(X_test, y_test):.4f}")

# Feature importance
xgb.plot_importance(xgb_model)
plt.show()
```

### LightGBM

```python
import lightgbm as lgb

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42
)

lgb_model.fit(X_train, y_train)
print(f"LightGBM Accuracy: {lgb_model.score(X_test, y_test):.4f}")
```

### CatBoost

```python
from catboost import CatBoostClassifier

# CatBoost
cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    verbose=0,
    random_state=42
)

cat_model.fit(X_train, y_train)
print(f"CatBoost Accuracy: {cat_model.score(X_test, y_test):.4f}")
```

---

## 4. Ensemble Comparison

```python
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.model_selection import cross_val_score

# Compare models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(n_estimators=50),
    'XGBoost': xgb.XGBClassifier(n_estimators=100),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 5. Hyperparameter Tuning

### Grid Search for Random Forest

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
```

---

## 💻 Python Code Examples

```python
# === Complete Tree-Based Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate
print("Random Forest:")
print(classification_report(y_test, rf.predict(X_test)))

print("\nXGBoost:")
print(classification_report(y_test, xgb_model.predict(X_test)))

# Cross-Validation
rf_cv = cross_val_score(rf, X_train, y_train, cv=5)
xgb_cv = cross_val_score(xgb_model, X_train, y_train, cv=5)

print(f"\nRF CV: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")
print(f"XGB CV: {xgb_cv.mean():.4f} (+/- {xgb_cv.std():.4f})")
```

---

## 📊 Summary Tables

### Tree Algorithms Comparison

| Algorithm | Type | Speed | Accuracy | Use Case |
|-----------|------|-------|----------|----------|
| Decision Tree | Single | Fast | Low | Interpretability |
| Random Forest | Bagging | Medium | High | General purpose |
| XGBoost | Boosting | Medium | Very High | Competitions |
| LightGBM | Boosting | Fast | Very High | Large datasets |
| CatBoost | Boosting | Medium | Very High | Categorical features |

### Hyperparameters

| Parameter | Effect | Tuning Range |
|-----------|--------|--------------|
| n_estimators | More trees = better | 50-500 |
| max_depth | Tree complexity | 3-15 |
| learning_rate | Step size | 0.01-0.3 |
| min_samples_split | Overfitting control | 2-20 |

---

## 🎯 ML Applications

| Algorithm | ML Application |
|-----------|----------------|
| Decision Tree | Rule extraction |
| Random Forest | Feature importance |
| XGBoost | Kaggle competitions |
| LightGBM | Large-scale ranking |
| CatBoost | Categorical data |

---

**Status:** ✅ Complete
**Next:** Instance-Based Learning
