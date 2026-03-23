# ML Fundamentals - Practice Problems

## Topic 1: Introduction to ML

### Level 1: Basic

**1.1** Identify the type of learning:
```
1. Predicting house prices
2. Grouping customers by behavior
3. Learning to play chess through rewards
4. Classifying emails as spam
```

**1.2** Match the problem type:
```
1. Regression
2. Classification
3. Clustering
4. Dimensionality Reduction

a) Customer segmentation
b) House price prediction
c) PCA
d) Spam detection
```

---

## Topic 2: Supervised Learning

### Level 1: Basic

**2.1** Implement Linear Regression:
```python
from sklearn.linear_model import LinearRegression

# 1. Create model
# 2. Fit on training data
# 3. Predict on test data
# 4. Calculate R² score
```

**2.2** Classification Metrics:
```python
# Given y_true and y_pred:
# 1. Calculate accuracy, precision, recall, F1
# 2. Print classification report
```

### Level 2: Intermediate

**2.3** Compare Regression vs Classification:
```python
# Create both regression and classification datasets
# Train appropriate models
# Evaluate with correct metrics
```

---

## Topic 3: Unsupervised Learning

### Level 2: Intermediate

**3.1** K-Means Clustering:
```python
from sklearn.cluster import KMeans

# 1. Fit K-Means with k=3
# 2. Get cluster labels
# 3. Calculate silhouette score
# 4. Find optimal k using elbow method
```

**3.2** PCA:
```python
from sklearn.decomposition import PCA

# 1. Fit PCA to retain 95% variance
# 2. Transform data
# 3. Get explained variance ratio
```

---

## Topic 4: Model Evaluation

### Level 2: Intermediate

**4.1** Cross-Validation:
```python
from sklearn.model_selection import cross_val_score, KFold

# 1. Perform 5-fold CV
# 2. Calculate mean and std of scores
# 3. Compare with single train-test split
```

**4.2** Hyperparameter Tuning:
```python
from sklearn.model_selection import GridSearchCV

# 1. Define parameter grid
# 2. Create GridSearchCV
# 3. Fit on training data
# 4. Get best parameters and score
```

---

## Topic 5: Feature Engineering

### Level 2: Intermediate

**5.1** Feature Selection:
```python
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Select top 10 features using F-test
# 2. Get feature scores
# 3. Compare model with selected vs all features
```

**5.2** Feature Scaling:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Standardize features
# 2. Normalize to [0, 1]
# 3. Compare model performance with different scalers
```

---

## Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1 Learning Types
```
1. Supervised (Regression)
2. Unsupervised (Clustering)
3. Reinforcement Learning
4. Supervised (Classification)
```

### 2.1 Linear Regression
```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
```

### 3.1 K-Means
```python
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
silhouette = silhouette_score(X, labels)

# Elbow method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
```

### 4.1 Cross-Validation
```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)
print(f"CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 5.1 Feature Selection
```python
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ ML Fundamentals Complete!
