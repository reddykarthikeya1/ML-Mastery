# 6.1-6.5 ML Fundamentals - Complete Reference

## 🎯 Quick Overview
- **Machine Learning**: Algorithms that learn from data
- **Supervised Learning**: Learn from labeled data
- **Unsupervised Learning**: Find patterns in unlabeled data
- **Model Evaluation**: Assess model performance
- **Feature Engineering**: Create better features
- **Foundation for**: All ML and Deep Learning

---

## Part 1: Introduction to ML (6.1)

### What is Machine Learning?

```
ML = Algorithms that improve through experience

Traditional Programming:
Input + Program → Output

Machine Learning:
Input + Output → Model (Program)
```

### Types of Learning

```
Supervised Learning:
- Input-output pairs
- Learn mapping function
- Examples: Classification, Regression

Unsupervised Learning:
- Only input data
- Find hidden patterns
- Examples: Clustering, Dimensionality Reduction

Reinforcement Learning:
- Learn through rewards/penalties
- Agent-environment interaction
- Examples: Game playing, Robotics

Semi-supervised:
- Small labeled + large unlabeled data

Self-supervised:
- Create labels from data itself
```

### ML Workflow

```
1. Problem Definition
2. Data Collection
3. Data Preprocessing
4. Feature Engineering
5. Model Selection
6. Training
7. Evaluation
8. Hyperparameter Tuning
9. Deployment
10. Monitoring
```

---

## Part 2: Supervised Learning (6.2)

### Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Regularized Regression
ridge = Ridge(alpha=1.0)  # L2 regularization
lasso = Lasso(alpha=1.0)  # L1 regularization
```

### Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba[:, 1])

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# TN FP
# FN TP
```

### Model Comparison

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| Linear Regression | Continuous target | Simple, interpretable | Assumes linearity |
| Logistic Regression | Binary classification | Fast, probabilistic | Linear boundary |
| Decision Tree | Any | Interpretable, no scaling | Overfits easily |
| Random Forest | Any | Accurate, robust | Less interpretable |
| SVM | Small-medium data | Effective in high-D | Slow on large data |
| XGBoost | Tabular data | State-of-the-art | Many hyperparameters |

---

## Part 3: Unsupervised Learning (6.3)

### Clustering

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Elbow method for optimal k
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# DBSCAN (density-based)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3)
labels = hc.fit_predict(X)

# Evaluation
silhouette = silhouette_score(X, labels)
db_score = davies_bouldin_score(X, labels)
```

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_var = pca.explained_variance_ratio_

# t-SNE (visualization)
tsne = TSNE(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X)

# UMAP (faster than t-SNE)
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X)
```

---

## Part 4: Model Evaluation (6.4)

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut

# K-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Stratified K-Fold (for classification)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold)

# Leave-One-Out CV
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

### Bias-Variance Tradeoff

```
Bias: Error from erroneous assumptions
- High bias → Underfitting
- Model too simple

Variance: Error from sensitivity to training data
- High variance → Overfitting
- Model too complex

Total Error = Bias² + Variance + Irreducible Error

Solutions:
- Underfitting: More features, less regularization
- Overfitting: More data, regularization, simpler model
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

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

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

# Grid Search
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
best_params = grid_search.best_params_

# Random Search
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [5, 10, 15, None],
    'min_samples_split': uniform(0.01, 0.2)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
```

---

## Part 5: Feature Engineering (6.5)

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Filter Methods
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Mutual Information
mi_scores = mutual_info_classif(X, y)

# Wrapper Methods (RFE)
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X, y)
selected_features = rfe.support_

# Embedded Methods (Feature Importance)
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

### Feature Extraction

```python
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)

# Text Features (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(texts)

# Image Features (using pre-trained CNN)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, extract_features

model = VGG16(weights='imagenet', include_top=False)
img = image.load_img('image.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
```

### Feature Transformation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Power Transformation (normalize variance)
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

# Log Transform
X_log = np.log1p(X)

# Binning
X_binned = pd.cut(X['age'], bins=[0, 18, 35, 65, 100], 
                  labels=['Young', 'Adult', 'Middle', 'Senior'])
```

### Handling Categorical Variables

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = ohe.fit_transform(X_categorical)

# Label Encoding (ordinal)
le = LabelEncoder()
X_labeled = le.fit_transform(X_ordinal)

# Ordinal Encoding
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_ordinal = oe.fit_transform(X)

# Pandas get_dummies
X_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')
```

---

## 💻 Python Code Examples

```python
# === Complete ML Pipeline Example ===

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Feature importance
if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = pipeline.named_steps['classifier'].feature_importances_
    for feature, importance in zip(X.columns, importances):
        print(f"{feature}: {importance:.4f}")

# === Hyperparameter Tuning Pipeline ===

from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.3f}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## 📊 Summary Tables

### ML Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| Linear Regression | Supervised | Regression |
| Logistic Regression | Supervised | Binary Classification |
| Decision Tree | Supervised | Classification/Regression |
| Random Forest | Supervised | Classification/Regression |
| SVM | Supervised | Classification |
| K-Means | Unsupervised | Clustering |
| DBSCAN | Unsupervised | Density-based Clustering |
| PCA | Unsupervised | Dimensionality Reduction |

### Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| Precision | TP/(TP+FP) | Minimize false positives |
| Recall | TP/(TP+FN) | Minimize false negatives |
| F1 Score | 2×(Precision×Recall)/(Precision+Recall) | Balance precision/recall |
| ROC-AUC | Area under ROC curve | Binary classification |
| RMSE | √(MSE) | Regression |
| R² | 1 - SS_res/SS_tot | Regression fit |

### Cross-Validation Methods

| Method | Folds | Use Case |
|--------|-------|----------|
| K-Fold | k | General purpose |
| Stratified K-Fold | k | Imbalanced classification |
| Leave-One-Out | n-1 | Very small datasets |
| Time Series Split | k | Time series data |

---

## 🎯 ML Applications

| Concept | Real-World Application |
|---------|----------------------|
| Regression | House price prediction |
| Classification | Spam detection, Disease diagnosis |
| Clustering | Customer segmentation |
| Dimensionality Reduction | Visualization, Feature compression |
| Feature Engineering | Improving model performance |

---

**Status:** ✅ Complete
**Next:** Phase 3 - Core ML & Deep Learning
