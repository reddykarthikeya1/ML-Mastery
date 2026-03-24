# 6.1 Introduction to Machine Learning

## 🎯 Quick Overview
- **Machine Learning**: Algorithms that learn from data
- **Types**: Supervised, Unsupervised, Reinforcement
- **Workflow**: From data to deployment
- **Foundation for**: All ML and Deep Learning

---

## 1. What is Machine Learning?

### Definition

```
Machine Learning = Algorithms that improve through experience

Arthur Samuel (1959): "Field of study that gives computers the ability to learn without being explicitly programmed"

Tom Mitchell (1997): "A computer program learns from experience E with respect to task T and performance measure P, if its performance on T, as measured by P, improves with experience E"
```

### Traditional Programming vs ML

```
Traditional Programming:
Input + Program (Rules) → Output

Machine Learning:
Input + Output → Model (Rules)
```

### Types of Learning

```
Supervised Learning:
- Input: Labeled training data (X, y)
- Output: Function mapping X → y
- Tasks: Classification, Regression
- Examples: Spam detection, House price prediction

Unsupervised Learning:
- Input: Unlabeled data (X)
- Output: Hidden patterns/structure
- Tasks: Clustering, Dimensionality Reduction
- Examples: Customer segmentation, PCA

Reinforcement Learning:
- Input: Environment, rewards/penalties
- Output: Optimal policy
- Tasks: Decision making
- Examples: Game playing, Robotics

Semi-Supervised Learning:
- Small labeled + large unlabeled data
- Example: Medical imaging

Self-Supervised Learning:
- Create labels from data itself
- Example: Language models (BERT, GPT)
```

---

#### 🧒 ELI5: Types of Machine Learning

> Imagine different ways a student can learn.
>
> **Supervised Learning** (Teacher with answer key):
> - Teacher gives problems WITH answers: "2+2=4, 3+3=6"
> - Student learns the pattern
> - Test: "What's 5+5?" → Student knows it's 10!
> - Like: Learning with labeled training data
>
> **Unsupervised Learning** (No teacher, just data):
> - Student gets a box of mixed LEGO: red, blue, squares, circles
> - No instructions, but student notices: "Hmm, reds are mostly squares"
> - Creates own groups/clusters
> - Like: Finding hidden patterns in unlabeled data
>
> **Reinforcement Learning** (Learning by trial and error):
> - Student plays a video game
> - Die → lose points (bad action!)
> - Beat level → gain points (good action!)
> - Learns which actions lead to rewards
> - Like: Training AI to play games or drive cars
>
> **Semi-Supervised** (Few answers, lots of practice):
> - Teacher gives 10 problems with answers
> - Plus 1000 problems WITHOUT answers
> - Student uses the 10 to figure out the rest
> - Like: Medical diagnosis (few labeled scans, many unlabeled)
>
> **Self-Supervised** (Making your own homework):
> - Student covers up words in a sentence: "The cat sat on the ___"
> - Tries to guess: "mat"? "chair"?
> - Checks by uncovering: "mat" - correct!
> - Like: How BERT/GPT learn from text without labels

</details>

---

## 2. ML Problem Types

### Regression Problems

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Predict continuous value
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

### Classification Problems

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Binary classification
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Multi-class classification
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Multi-class metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

### Clustering Problems

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Group similar items
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Metrics
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette:.4f}")
```

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA

# Compress data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
```

---

## 3. ML Workflow

### Complete Pipeline

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# 1. Problem Definition
# - What are we trying to predict?
# - What metrics matter?

# 2. Data Collection
data = load_breast_cancer()
X, y = data.data, data.target

# 3. Data Preprocessing
# - Handle missing values
# - Remove duplicates
# - Data cleaning

# 4. Feature Engineering
# - Feature selection
# - Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Model Selection
model = RandomForestClassifier(random_state=42)

# 7. Training
model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 9. Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

# 10. Deployment
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

## 4. ML Applications

### Real-World Applications

```
Healthcare:
- Disease diagnosis
- Drug discovery
- Medical image analysis

Finance:
- Fraud detection
- Algorithmic trading
- Credit scoring

Technology:
- Search engines
- Recommendation systems
- Speech recognition

Transportation:
- Autonomous vehicles
- Route optimization
- Traffic prediction

Retail:
- Demand forecasting
- Customer segmentation
- Price optimization
```

---

## 💻 Python Code Examples

```python
# === Complete ML Pipeline Example ===

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15, n_redundant=5,
                           random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-Validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(12, 6))
plt.bar(range(10), importances[indices])
plt.xlabel('Feature Rank')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()
```

---

## 📊 Summary Tables

### Types of Learning

| Type | Input | Output | Example |
|------|-------|--------|---------|
| Supervised | Labeled data | Predictions | Spam detection |
| Unsupervised | Unlabeled data | Patterns | Customer segmentation |
| Reinforcement | Environment | Policy | Game playing |
| Semi-supervised | Mixed data | Predictions | Medical imaging |

### ML Workflow Steps

| Step | Purpose | Tools |
|------|---------|-------|
| Data Collection | Gather data | APIs, Databases |
| Preprocessing | Clean data | Pandas, NumPy |
| Feature Engineering | Create features | Scikit-learn |
| Model Selection | Choose algorithm | Scikit-learn |
| Training | Learn patterns | ML libraries |
| Evaluation | Measure performance | Metrics |
| Tuning | Optimize | GridSearchCV |
| Deployment | Put in production | Flask, Docker |

### Problem Types

| Type | Output | Metrics | Example |
|------|--------|---------|---------|
| Regression | Continuous | RMSE, R² | House prices |
| Classification | Category | Accuracy, F1 | Spam detection |
| Clustering | Groups | Silhouette | Segmentation |
| Dimensionality Reduction | Compressed features | Explained variance | PCA |

---

## 🎯 ML Applications

| Industry | Application | Algorithm |
|----------|-------------|-----------|
| Healthcare | Disease diagnosis | Random Forest |
| Finance | Fraud detection | Logistic Regression |
| Retail | Recommendation | Collaborative Filtering |
| Tech | Search ranking | Gradient Boosting |
| Auto | Self-driving | Deep Learning |

---

---

## ❓ Quick Check Questions

1. What is the difference between "Traditional Programming" and "Machine Learning"?
2. What are the three main types of ML?
3. What is the difference between classification and regression?
4. Why do we need a validation set in addition to train and test sets?
5. What is data leakage?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Traditional Programming:** You write the rules (code) and provide data to get an output. **Machine Learning:** You provide the data and the expected output, and the algorithm learns the rules (patterns) itself.
2. **Supervised** (labeled data), **Unsupervised** (unlabeled data), and **Reinforcement Learning** (reward-based).
3. **Classification** predicts a category/label (discrete). **Regression** predicts a numerical value (continuous).
4. The **Validation set** is used to tune hyperparameters and select the best model, while the **Test set** is kept completely separate to provide a final, unbiased evaluation of the chosen model.
5. **Data leakage** occurs when information from outside the training dataset (typically from the test or validation set) is used to train the model, leading to overly optimistic but false performance results.

</details>

---

**Status:** ✅ Complete
**Next:** Supervised Learning
