# 6.5 Feature Engineering

## 🎯 Quick Overview
- **Feature Selection**: Choose relevant features
- **Feature Extraction**: Transform to lower dimensions
- **Feature Creation**: Create new features
- **Foundation for**: Improved model performance

---

## 1. Feature Selection

### Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select K best features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected features
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
```

### Wrapper Methods

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

# RFE
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
rfe.fit(X, y)

# RFECV
rfecv = RFECV(estimator=RandomForestClassifier(), cv=5)
rfecv.fit(X, y)

print(f"Optimal features: {rfecv.n_features_}")
```

### Embedded Methods

```python
from sklearn.feature_selection import SelectFromModel

# Feature importance
model = RandomForestClassifier()
model.fit(X, y)

# Select features
selector = SelectFromModel(model, threshold='mean')
X_selected = selector.fit_transform(X, y)
```

---

## 2. Feature Extraction

### PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)

print(f"Original: {X.shape[1]}, PCA: {X_pca.shape[1]}")
```

### LDA

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)
```

---

## 3. Feature Transformation

### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
```

### Encoding

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-Hot Encoding
ohe = OneHotEncoder()
X_ohe = ohe.fit_transform(X_categorical)

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

---

## 💻 Python Code Examples

```python
# === Complete Feature Engineering Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=15)),
    ('classifier', RandomForestClassifier())
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
```

---

## 📊 Summary Tables

### Feature Selection Methods

| Method | Type | Pros | Cons |
|--------|------|------|------|
| Filter | Univariate | Fast | Ignores interactions |
| RFE | Wrapper | Considers interactions | Slow |
| Feature Importance | Embedded | Fast | Biased to high cardinality |

### Feature Scaling

| Method | Formula | Use Case |
|--------|---------|----------|
| Standard | (x - mean) / std | Most algorithms |
| MinMax | (x - min) / (max - min) | Neural networks |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Feature Selection | Dimensionality reduction |
| PCA | Visualization, compression |
| Encoding | Categorical variables |

---

**Status:** ✅ Complete
**Next:** Classical ML Algorithms
