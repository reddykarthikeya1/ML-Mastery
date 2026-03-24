# 7.2 Tree-Based Models

## 🎯 Quick Overview
- **Decision Trees**: Interpretable, non-linear
- **Random Forest**: Ensemble of trees
- **Boosting**: Sequential improvement
- **XGBoost/LightGBM/CatBoost**: Modern implementations
- **Foundation for**: Most accurate tabular ML

---

#### 🧒 ELI5: Decision Trees, Random Forest & Boosting

> Imagine playing "20 Questions" to guess what animal I'm thinking of.
>
> **Decision Tree** (One person asking questions):
>
> Question 1: "Does it fly?"
> - YES → Question 2: "Does it have feathers?"
>   - YES → "It's a bird!"
>   - NO → "It's a bat!"
> - NO → Question 2: "Does it live in water?"
>   - YES → "It's a fish!"
>   - NO → "It's a dog!"
>
> **How tree learns**:
> - Looks at ALL animals
> - Finds BEST first question (most separating)
> - "Does it fly?" splits animals better than "Is it red?"
> - Keeps splitting until each group is one type
>
> **Entropy/Gini** (How mixed up is the group?):
> - Basket of 10 apples: Pure! (Gini = 0)
> - Basket of 5 apples + 5 oranges: Mixed! (Gini = 0.5)
> - Tree wants: Pure baskets at the end!
>
> **Random Forest** (Committee of trees):
>
> One tree: "I think it's a bird" (might be wrong!)
> 100 trees: 
> - Tree 1: "Bird!" (saw it has wings)
> - Tree 2: "Bird!" (saw it has beak)
> - Tree 3: "Actually... bat?" (only looked at flying)
> - ...
> - Vote: 87 say bird, 13 say bat → "It's a BIRD!"
>
> **Why Random Forest is better**:
> - Each tree sees DIFFERENT subset of data
> - Each tree considers DIFFERENT features
> - Trees are diverse (uncorrelated)
> - Together: Much smarter than any one tree!
> - Like: Crowd wisdom
>
> **Boosting** (Sequential improvement):
>
> **Round 1**: Tree makes predictions
> - Gets 80% right, 20% wrong
>
> **Round 2**: NEW tree focuses on the 20% wrong!
> - "I'll fix the mistakes from Round 1"
>
> **Round 3**: ANOTHER tree fixes remaining mistakes
>
> **Final**: Combine all trees (weighted by accuracy)
> - Like: Team where each member fixes previous mistakes!
>
> **Bagging vs Boosting**:
> - **Bagging** (Random Forest): Trees work INDEPENDENTLY, then vote
> - **Boosting** (XGBoost): Trees work SEQUENTIALLY, each fixes errors

</details>

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

#### 🧒 ELI5: Ensemble Methods - Bagging, Boosting, Stacking, Voting

> Imagine you need to make an important decision. Should you go alone or ask for advice?
>
> **Single Model** (Going alone):
> - One person's opinion
> - Might be wrong!
> - "I think we should invest in stocks"
>
> **Ensemble** (Asking a group):
> - Many people's opinions combined
> - More reliable!
> - "8 out of 10 advisors say invest in stocks"
>
> **Bagging** (Bootstrap Aggregating - Parallel voting):
>
> **How it works**:
> - Ask 100 advisors INDEPENDENTLY
> - Each advisor sees DIFFERENT subset of data
> - Advisor 1: Sees data from 2010-2015
> - Advisor 2: Sees data from 2012-2017
> - ...
> - Final decision: MAJORITY VOTE!
>
> **Random Forest** = Bagging + Decision Trees:
> - 100 trees, each trained on random subset
> - Each tree also sees RANDOM subset of features!
> - Tree 1: Uses age + income
> - Tree 2: Uses income + location
> - Tree 3: Uses age + location
> - More diversity = better ensemble!
>
> **Why Bagging works**:
> - Reduces variance (averaging cancels out errors)
> - Like: "Ask 100 people, average their answers"
> - Individual might be wrong, group is usually right!
>
> **Boosting** (Sequential improvement):
>
> **How it works**:
> - Advisor 1: "Invest in stocks" → 60% right
> - Advisor 2: "Focus on cases Advisor 1 got WRONG"
> - Advisor 3: "Focus on cases 1 & 2 got WRONG"
> - Each advisor learns from previous mistakes!
>
> **AdaBoost** (Adaptive Boosting):
> - Misclassified points get HIGHER weight
> - Next model focuses MORE on hard cases
> - Like: "Students who failed get extra tutoring"
>
> **Gradient Boosting** (Fitting residuals):
> - Model 1: Predicts 100, Actual is 150 → Error is 50
> - Model 2: Predicts the ERROR (50)
> - Model 3: Predicts remaining error
> - Final: Sum of all models!
>
> **XGBoost** (Extreme Gradient Boosting):
> - "Gradient Boosting but optimized"
> - Faster, more accurate
> - Regularization (doesn't overfit)
> - Handles missing data
> - Like: "Boosting on steroids"
>
> **LightGBM** (Light Gradient Boosting):
> - "XGBoost but faster"
> - Uses histogram bins (less memory)
> - Grows trees leaf-wise (not level-wise)
> - Better for large datasets
>
> **CatBoost** (Categorical Boosting):
> - "Best for categorical data"
> - Handles categories automatically
> - No need for one-hot encoding!
> - "City: NY, LA, SF" → understands directly
>
> **Stacking** (Meta-learning):
>
> **Level 1** (Base models):
> - Model 1 (Linear): "Stocks will go up"
> - Model 2 (Tree): "Stocks will go down"
> - Model 3 (SVM): "Stocks will stay flat"
>
> **Level 2** (Meta-model):
> - Takes predictions from Level 1 as INPUT
> - Learns: "When Linear and Tree disagree, trust Linear"
> - "When all three agree, they're usually right"
> - Meta-model learns which models to trust!
>
> **Like**: A manager who knows:
> - "Alice is good at predicting tech stocks"
> - "Bob is good at predicting market crashes"
> - Combines their predictions intelligently!
>
> **Voting** (Simple ensemble):
>
> **Hard Voting** (Majority rules):
> - Model 1: "YES"
> - Model 2: "YES"
> - Model 3: "NO"
> - Final: "YES" (2 out of 3)
>
> **Soft Voting** (Weighted average):
> - Model 1: "80% YES"
> - Model 2: "60% YES"
> - Model 3: "30% YES"
> - Average: 56.7% YES → Final: "YES"
>
> **When to use which**:
> - **Bagging**: High variance models (deep trees)
> - **Boosting**: Need maximum accuracy, have time
> - **Stacking**: Have diverse models, want best performance
> - **Voting**: Quick ensemble, baseline improvement
>
> **Bagging vs Boosting**:
> - **Bagging**: Parallel (all at once), reduces variance
> - **Boosting**: Sequential (one after another), reduces bias

</details>

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

## ❓ Quick Check Questions

1. What is the difference between "Bagging" (used in Random Forest) and "Boosting" (used in XGBoost)?
2. What are the two main splitting criteria for Decision Trees, and how do they differ?
3. What is the "Out-of-Bag" (OOB) error in Random Forest?
4. How does a Gradient Boosting machine improve its predictions at each step?
5. Why is "pruning" important for a single Decision Tree?

---

## 📝 Answers to Quick Check

1. **Bagging** builds multiple independent trees in parallel using different bootstrap samples and averages their results to reduce variance. **Boosting** builds trees sequentially, where each new tree tries to correct the errors made by the previous trees, focusing on bias reduction.
2. The two main criteria are **Gini Impurity** and **Entropy (Information Gain)**. Gini is computationally faster as it doesn't involve logarithms, while Entropy is slightly more theoretically sound and often produces similar results.
3. **OOB error** is the mean prediction error on each training sample $x_i$, using only the trees that did not have $x_i$ in their bootstrap sample. It provides an unbiased estimate of the test error without needing a separate validation set.
4. **Gradient Boosting** improves by fitting a new weak learner (usually a small decision tree) to the **negative gradient** (residuals) of the loss function of the current ensemble.
5. **Pruning** (setting `max_depth` or `min_samples_leaf`) is crucial to prevent the tree from becoming overly complex and "memorizing" the training data, which leads to **overfitting** and poor generalization to new data.

---

**Status:** ✅ Complete
**Next:** Instance-Based Learning
