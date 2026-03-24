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

#### 🧒 ELI5: Cross-Validation (K-Fold, LOOCV)

> Imagine you have ONE textbook and 5 students need to study from it.
>
> **Problem**: If you give the whole book to 1 student, others can't study!
>
> **K-Fold Cross-Validation** (Sharing the textbook):
> 1. Tear book into 5 equal parts (5 folds)
> 2. Round 1: Student 1 studies parts 2-5, gets tested on part 1
> 3. Round 2: Student 2 studies parts 1,3-5, gets tested on part 2
> 4. Continue for all 5 students...
> 5. Average all 5 test scores = TRUE understanding!
>
> **Why not just ONE train/test split?**:
> - What if you got unlucky and test has ONLY hard questions?
> - Or training has ONLY easy examples?
> - K-Fold: EVERY page gets used for testing ONCE
> - More reliable estimate!
>
> **Leave-One-Out CV (LOOCV)** - Extreme sharing:
> - Book has 100 pages
> - Round 1: Study 99 pages, test on page 1
> - Round 2: Study 99 pages, test on page 2
> - ...100 rounds!
> - Super accurate but SLOW (100 training sessions!)
>
> **When to use**:
> - **5-Fold or 10-Fold**: Standard, good balance
> - **LOOCV**: Tiny dataset (every example counts!)
> - **Stratified K-Fold**: Unbalanced classes (keep same % in each fold)

</details>

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

---

#### 🧒 ELI5: Bias-Variance Tradeoff, Overfitting & Underfitting

> Imagine you're studying for a math test.
>
> **High Bias (Underfitting)** - The Student Who Didn't Study Enough:
> - Learns ONLY: "Addition exists"
> - Test question: "What's 2+2?" → "Hmm, I know it's some math... 1?"
> - Too simple! Doesn't know enough to answer correctly
> - **Fix**: Study more (add more features, use complex model)
>
> **High Variance (Overfitting)** - The Student Who Memorized Everything:
> - Memorizes EVERY practice problem: "2+2=4, 3+5=8, 7+9=16"
> - Test question: "What's 4+6?" → Never saw THIS exact problem! PANIC!
> - Too complex! Memorized instead of understanding
> - **Fix**: Understand concepts, not just memorize (regularization, simpler model)
>
> **Good Balance** - The Student Who Understands Concepts:
> - Learns the RULE: "Addition combines numbers"
> - Practices with DIFFERENT problems
> - Test question: "What's 4+6?" → "I understand addition, it's 10!"
> - Works on problems NEVER seen before!
>
> **The Tradeoff**:
> - Simple model → High bias, low variance (underfits)
> - Complex model → Low bias, high variance (overfits)
> - Just right → Low bias, low variance (generalizes well)
>
> **Learning Curves Explained**:
> - **Training score**: How well model does on practice problems
> - **Validation score**: How well model does on NEW test problems
> - **Gap between them**: How much is it overfitting?
> - Small gap = Good! Large gap = Overfitting!

</details>

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

---

#### 🧒 ELI5: Hyperparameter Tuning - Grid Search, Random Search, Bayesian Optimization

> Imagine you're baking a cake and need to find the perfect recipe.
>
> **Hyperparameters** (Recipe settings):
> - Flour: 2 cups, 3 cups, or 4 cups?
> - Sugar: 1 cup or 1.5 cups?
> - Bake time: 30 min, 45 min, or 60 min?
> - Oven temp: 350°F or 375°F?
>
> **These are HYPERparameters** (set BEFORE baking/training)
> NOT regular parameters (amount of batter = learned during training)
>
> **Grid Search** (Try EVERY combination):
>
> **The approach**:
> - Flour: [2, 3, 4] cups
> - Sugar: [1, 1.5] cups
> - Time: [30, 45, 60] min
>
> Try ALL combinations:
> 1. 2 cups flour + 1 cup sugar + 30 min
> 2. 2 cups flour + 1 cup sugar + 45 min
> 3. 2 cups flour + 1 cup sugar + 60 min
> 4. 2 cups flour + 1.5 cups sugar + 30 min
> ... (continues for ALL 3×2×3 = 18 combinations!)
>
> **Pros**: Guaranteed to find best combo
> **Cons**: SLOW! (What if you have 10 ingredients with 10 options each? = 10¹⁰ = 10 BILLION cakes!)
>
> **Random Search** (Try RANDOM combinations):
>
> **The approach**:
> - "I'll try 50 random recipes"
> - Recipe 1: 2.3 cups flour, 1.1 cups sugar, 37 min (random!)
> - Recipe 2: 3.7 cups flour, 1.4 cups sugar, 52 min (random!)
> - ...
>
> **Pros**: Much faster!
> **Cons**: Might miss the perfect combo
>
> **Why Random often beats Grid**:
> - Some ingredients don't matter much (sugar between 1-1.5 cups = same)
> - Grid wastes time on unimportant settings
> - Random explores MORE of the important settings!
>
> **Bayesian Optimization** (Smart guessing):
>
> **The approach**:
> - Try recipe 1: "Too dry"
> - Think: "Need more moisture → more sugar OR less time"
> - Try recipe 2 (smart choice based on recipe 1!)
> - "Better! But too sweet"
> - Try recipe 3 (even smarter!): "Less sugar, same time"
>
> **Like**: Having a intuition that gets better with each try!
>
> **Bayesian Optimization uses**:
> - "Surrogate model" (predicts how good untried recipes are)
> - "Acquisition function" (decides which recipe to try next)
>
> **Pros**: Finds best with FEWEST tries!
> **Cons**: More complex to implement
>
> **When to use which**:
> - **Grid Search**: 2-3 hyperparameters, small ranges
> - **Random Search**: 4-10 hyperparameters (default choice!)
> - **Bayesian**: Expensive training (each try takes hours/days)
>
> **Real example**:
> - Training neural network takes 1 day
> - Grid Search: 100 combinations = 100 days! 😱
> - Bayesian: Finds good combo in 10-15 tries = 15 days! ✅

</details>

---

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

---

## ❓ Quick Check Questions

1. What is K-Fold Cross-Validation?
2. What is the difference between Bias and Variance?
3. How do you detect overfitting using a learning curve?
4. What is the difference between Grid Search and Random Search?
5. Why use Stratified K-Fold instead of regular K-Fold?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **K-Fold Cross-Validation** is a technique where the dataset is split into $K$ subsets (folds). The model is trained $K$ times, each time using a different fold as the test set and the remaining $K-1$ folds as the training set.
2. **Bias** is error from overly simple assumptions (underfitting). **Variance** is error from high sensitivity to small fluctuations in the training set (overfitting).
3. **Overfitting** is detected when there is a large gap between the training performance (very high) and the validation performance (significantly lower).
4. **Grid Search** tries every possible combination of hyperparameters in a specified range. **Random Search** samples a fixed number of random combinations from the range, which is often faster and just as effective.
5. **Stratified K-Fold** ensures that each fold has the same proportion of class labels as the entire dataset, which is critical for maintaining consistency in classification tasks with imbalanced classes.

</details>

---

**Status:** ✅ Complete
**Next:** Feature Engineering
