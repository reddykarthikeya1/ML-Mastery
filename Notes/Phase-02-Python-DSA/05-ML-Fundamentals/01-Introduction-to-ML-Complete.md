# 6.1 Introduction to Machine Learning

## 🎯 Quick Overview
- **Machine Learning**: Algorithms that learn from data
- **Types of Learning**: Supervised, Unsupervised, Reinforcement
- **ML Workflow**: From data to deployment
- **Foundation for**: All ML and Deep Learning topics

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

### When to Use ML

```
Use ML when:
✓ Rules are too complex to code manually
✓ Patterns exist in data
✓ Data is abundant
✓ Task requires adaptation

Don't use ML when:
✗ Simple rules suffice
✗ No data available
✗ Interpretability is critical
✗ Safety-critical with no room for error
```

---

## 2. Types of Learning

### Supervised Learning

```
Input: Labeled training data (X, y)
Output: Function mapping X → y

Tasks:
- Classification: Predict category
  - Email spam detection
  - Image classification
  - Disease diagnosis

- Regression: Predict continuous value
  - House price prediction
  - Stock price forecasting
  - Temperature prediction

Algorithms:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- SVM
- Neural Networks
```

### Unsupervised Learning

```
Input: Unlabeled data (X)
Output: Hidden patterns/structure

Tasks:
- Clustering: Group similar items
  - Customer segmentation
  - Document clustering
  - Image segmentation

- Dimensionality Reduction: Compress data
  - Feature compression
  - Data visualization
  - Noise reduction

- Association: Find rules
  - Market basket analysis
  - Recommendation systems

Algorithms:
- K-Means
- Hierarchical Clustering
- DBSCAN
- PCA
- Autoencoders
```

### Reinforcement Learning

```
Input: Environment, rewards/penalties
Output: Optimal policy (action selection)

Agent learns through:
- Trial and error
- Rewards for good actions
- Penalties for bad actions

Applications:
- Game playing (Chess, Go)
- Robotics
- Autonomous driving
- Resource management

Algorithms:
- Q-Learning
- Policy Gradient
- Actor-Critic
- Deep Q-Networks (DQN)
```

### Semi-Supervised Learning

```
Input: Small labeled + large unlabeled data
Output: Improved model

Use when:
- Labeling is expensive
- Abundant unlabeled data available

Applications:
- Medical imaging
- Speech recognition
- Web content classification
```

### Self-Supervised Learning

```
Create labels from data itself

Approach:
- Predict missing parts
- Predict future frames
- Predict context

Applications:
- Language models (BERT, GPT)
- Image representation learning
- Video prediction
```

---

## 3. ML Workflow

### Complete Pipeline

```
1. Problem Definition
   ↓
2. Data Collection
   ↓
3. Data Preprocessing
   ↓
4. Feature Engineering
   ↓
5. Model Selection
   ↓
6. Training
   ↓
7. Evaluation
   ↓
8. Hyperparameter Tuning
   ↓
9. Deployment
   ↓
10. Monitoring
```

### Step-by-Step Breakdown

```python
# 1. Problem Definition
# - What are we trying to predict?
# - What metrics matter?
# - What are the constraints?

# 2. Data Collection
# - Gather data from databases, APIs, files
# - Ensure data quality and quantity

# 3. Data Preprocessing
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
df = df.dropna()  # Handle missing
df = df.drop_duplicates()  # Remove duplicates

# 4. Feature Engineering
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Scale numeric features
scaler = StandardScaler()
df['scaled_feature'] = scaler.fit_transform(df[['feature']])

# Encode categorical
df = pd.get_dummies(df, columns=['category'])

# 5. Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# 6. Training
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Evaluation
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 8. Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 9. Deployment
import joblib

joblib.dump(best_model, 'model.pkl')

# 10. Monitoring
# - Track prediction drift
# - Monitor performance metrics
# - Retrain when needed
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

### Industry Examples

```python
# Example: Spam Detection
def detect_spam(email_text):
    """Classify email as spam or not"""
    # Features: word frequency, links, sender reputation
    # Model: Logistic Regression or Naive Bayes
    # Output: Probability of spam
    pass

# Example: House Price Prediction
def predict_house_price(features):
    """Predict house price from features"""
    # Features: size, location, bedrooms, age
    # Model: Linear Regression or Gradient Boosting
    # Output: Predicted price
    pass

# Example: Customer Churn Prediction
def predict_churn(customer_data):
    """Predict if customer will churn"""
    # Features: usage, complaints, tenure
    # Model: Random Forest or XGBoost
    # Output: Churn probability
    pass
```

---

## 5. ML Tools and Libraries

### Python ML Stack

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Deep Learning
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn

# Model evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Deployment
import joblib
import pickle
from flask import Flask, request, jsonify
```

---

## 💻 Python Code Examples

```python
# === Complete ML Pipeline Example ===

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importances = model.feature_importances_
for feature, importance in zip(iris.feature_names, importances):
    print(f"{feature}: {importance:.4f}")

# === Model Comparison ===

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")

# Plot comparison
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
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

1. What is the fundamental difference between Supervised and Unsupervised learning?
2. When should you use Reinforcement Learning?
3. List the high-level steps in a standard ML workflow.
4. Why is it critical to split your data into training and test sets?
5. What is the difference between classification and regression tasks?
6. When is semi-supervised learning particularly useful?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Supervised Learning** uses labeled data where the model learns to map inputs to a known target output. **Unsupervised Learning** uses unlabeled data to find hidden patterns or structures within the data itself.
2. **Reinforcement Learning** is used when an agent must learn to make a sequence of decisions by interacting with an environment to maximize a cumulative reward (e.g., games, robotics).
3. **ML Workflow:** Problem Definition → Data Collection → Data Preprocessing → Feature Engineering → Model Selection → Training → Evaluation → Deployment.
4. Splitting data ensures you can evaluate your model on **unseen data**, which provides an unbiased estimate of how the model will perform in the real world and helps detect overfitting.
5. **Classification** predicts a discrete category or class label (e.g., Spam vs. Not Spam). **Regression** predicts a continuous numerical value (e.g., predicting house prices).
6. **Semi-Supervised Learning** is useful when you have a small amount of labeled data and a large amount of unlabeled data, as it can use the unlabeled data to improve the learning process when manual labeling is expensive.

</details>

---

**Status:** ✅ Complete
**Next:** Supervised Learning
