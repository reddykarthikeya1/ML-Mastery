# 6.2 Supervised Learning

## 🎯 Quick Overview
- **Supervised Learning**: Learn from labeled data
- **Classification**: Predict categories
- **Regression**: Predict continuous values
- **Foundation for**: Most ML applications

---

## 1. Classification

### 🧠 Mathematical Intuition & Logic Behind Algorithms

**1. Logistic Regression**
*   **Logic:** Models the probability that an instance belongs to a specific class.
*   **Math:** Uses the Sigmoid function `σ(z) = 1 / (1 + e^(-z))` to squash linear combinations `(z = wX + b)` into a probability between 0 and 1.
*   **Cost Function:** Uses **Binary Cross-Entropy (Log Loss)** which heavily penalizes confident but wrong predictions: `Cost = -[y*log(p) + (1-y)*log(1-p)]`.
*   **Optimization:** Uses Gradient Descent to iteratively adjust weights `w` to minimize the cost.

**2. Decision Trees & Random Forests**
*   **Logic:** Recursively splits the dataset into smaller subsets based on feature values that best separate the classes.
*   **Math:** Uses criteria like **Gini Impurity** `1 - Σ(p_i)^2` or **Entropy** `-Σ(p_i * log_2(p_i))` to measure node purity.
*   **Splitting:** The feature with the highest **Information Gain** (reduction in impurity) is chosen for the root. Random Forests build hundreds of these trees on random subsets to reduce variance.

**3. Support Vector Machines (SVM)**
*   **Logic:** Finds the "hyperplane" that best separates classes by maximizing the **margin** (distance between the plane and the closest data points, known as support vectors).
*   **Math:** If data isn't linearly separable, it applies the **Kernel Trick** (e.g., RBF, Polynomial) to project data into higher dimensions where a linear hyperplane can separate them.

### Binary Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load data
X_train, X_test, y_train, y_test = load_data()

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
# TN FP
# FN TP
```

### Multi-Class Classification

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train
model = RandomForestClassifier()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
print(classification_report(y, y_pred, target_names=iris.target_names))

# Class probabilities
proba = model.predict_proba(X)
```

### Classification Algorithms

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy:.4f}")
```

---

## 2. Regression

### 🧠 Mathematical Intuition & Logic Behind Algorithms

**1. Linear Regression**
*   **Logic:** Fits a straight line (or hyperplane) through the data points that best represents the relationship between independent and dependent variables.
*   **Math:** `y = w_1*x_1 + w_2*x_2 + ... + b`
*   **Cost Function:** Minimizes the **Mean Squared Error (MSE)**, which is the sum of squared distances between predicted values and actual values.
*   **Optimization:** Uses Gradient Descent or the analytical Ordinary Least Squares (OLS) Normal Equation.

**2. Regularized Regression (Ridge & Lasso)**
*   **Logic:** Standard Linear Regression can overfit (weights become too large). Regularization adds a penalty to the loss function to keep weights small.
*   **Lasso (L1):** Adds the absolute value of magnitude of coefficients as penalty. Can shrink weights to exactly zero, performing automatic **feature selection**.
*   **Ridge (L2):** Adds squared magnitude of coefficients. Penalizes large weights but doesn't force them to zero. Good for handling multicollinearity (highly correlated features).

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Coefficients
print(f"Intercept: {model.intercept_:.4f}")
for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature}: {coef:.4f}")
```

### Regularized Regression

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# ElasticNet (L1 + L2)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)

# Compare
for name, model in [('Ridge', ridge), ('Lasso', lasso), ('ElasticNet', enet)]:
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R²: {r2:.4f}")
```

### Tree-Based Regression

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}")
```

---

## 3. Model Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, 
    precision_recall_curve, confusion_matrix,
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

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Precision-Recall Curve
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_proba)
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, explained_variance_score, 
    mean_absolute_percentage_error
)

# MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)

# R² Score (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

# Explained Variance
explained_var = explained_variance_score(y_test, y_pred)

# MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_test, y_pred)
```

---

## 4. Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Stratified Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

# For classification with imbalanced classes
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skfold, scoring='f1')

print(f"CV F1 Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Leave-One-Out CV

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

print(f"LOOCV Accuracy: {scores.mean():.4f}")
```

---

## 5. Handling Imbalanced Data

### Techniques

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE (Oversampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Undersampling
under = RandomUnderSampler(random_state=42)
X_under, y_under = under.fit_resample(X_train, y_train)

# Combined
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('under', RandomUnderSampler(random_state=42)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

### Class Weights

```python
# Automatic class weights
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Custom class weights
class_weights = {0: 1, 1: 10}  # Give more weight to minority class
model = LogisticRegression(class_weight=class_weights)
model.fit(X_train, y_train)
```

---

## 💻 Python Code Examples

```python
# === Complete Classification Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), data.feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

# === Regression Pipeline ===

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Load data
boston = load_boston()
X, y = boston.data, boston.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Ridge': Ridge(),
    'Gradient Boosting': GradientBoostingRegressor()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=5, scoring='r2')
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 📊 Summary Tables

### Classification Algorithms

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| Logistic Regression | Fast, interpretable | Linear boundary | Binary classification |
| Decision Tree | Interpretable, no scaling | Overfits | Small datasets |
| Random Forest | Accurate, robust | Less interpretable | General purpose |
| SVM | Effective in high-D | Slow on large data | Complex boundaries |
| KNN | Simple, no training | Slow prediction | Small datasets |

### Regression Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | avg((y-ŷ)²) | Lower is better |
| RMSE | √MSE | Same units as y |
| MAE | avg(\|y-ŷ\|) | Robust to outliers |
| R² | 1 - SS_res/SS_tot | 0-1, higher is better |

### Cross-Validation Methods

| Method | Folds | Use Case |
|--------|-------|----------|
| K-Fold | k | General purpose |
| Stratified | k | Imbalanced classification |
| LOOCV | n-1 | Very small datasets |
| Time Series | k | Time-dependent data |

---

## 🎯 ML Applications

| Task | Algorithm | Industry |
|------|-----------|----------|
| Spam Detection | Logistic Regression | Tech |
| Disease Diagnosis | Random Forest | Healthcare |
| House Pricing | Gradient Boosting | Real Estate |
| Customer Churn | XGBoost | Telecom |
| Credit Scoring | Logistic Regression | Finance |

---

---

## ❓ Quick Check Questions

1. How does Logistic Regression convert a linear combination of features into a probability?
2. What is the fundamental difference between Gini Impurity and Entropy in Decision Trees?
3. What are "Support Vectors" in an SVM model?
4. How does Lasso (L1) regularization perform automatic feature selection?
5. Why is the R² score (Coefficient of Determination) a preferred metric for regression over simple MSE?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. Logistic Regression uses the **Sigmoid function** ($\sigma(z) = 1 / (1 + e^{-z})$) to map the output of a linear equation (which can be any real number) into a range between 0 and 1, representing a probability.
2. Both measure node "purity," but **Gini Impurity** ($1 - \sum p_i^2$) is computationally faster because it doesn't involve logarithms, whereas **Entropy** ($-\sum p_i \log_2 p_i$) is derived from information theory and is slightly more sensitive to changes in class probabilities.
3. **Support Vectors** are the data points from the training set that lie closest to the decision boundary (hyperplane). They are the only points that actually determine the position and orientation of the boundary.
4. **Lasso (L1)** adds the absolute magnitude of coefficients as a penalty to the loss function. Because the L1 penalty is "diamond-shaped," the optimization often hits the axes, forcing some coefficients to become exactly zero, effectively removing those features.
5. **R²** is unitless and provides a standardized measure (0 to 1) of how well the independent variables explain the variance in the target. MSE is scale-dependent, making it harder to interpret without knowing the range of the target variable.

</details>

---

**Status:** ✅ Complete
**Next:** Unsupervised Learning
