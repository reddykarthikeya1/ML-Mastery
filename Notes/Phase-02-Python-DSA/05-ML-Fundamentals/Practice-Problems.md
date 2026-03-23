# ML Fundamentals - Practice Problems

## Topic 1: Introduction to ML

### Level 1: Basic

**1.1** Identify the type of learning:
```
1. Predicting house prices from features
2. Grouping customers by behavior
3. Learning to play chess through rewards
4. Classifying emails as spam/not spam
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
# 4. Calculate MSE and R²
```

### Level 2: Intermediate

**2.2** Classification Metrics:
```python
from sklearn.metrics import classification_report, confusion_matrix

# Given y_true and y_pred:
# 1. Calculate accuracy, precision, recall, F1
# 2. Print classification report
# 3. Print confusion matrix
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
# 4. Visualize first 2 components
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

**5.1** Feature Scaling:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Standardize features
# 2. Normalize to [0, 1]
# 3. Compare model performance with different scalers
```

**5.2** Feature Selection:
```python
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Select top 10 features using F-test
# 2. Get feature scores
# 3. Compare model with selected vs all features
```

---

## Topic 6: Complete ML Pipeline

### Level 3: Advanced

**6.1** End-to-End ML Project:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Complete pipeline:
# 1. Load and explore data
# 2. Split into train/test
# 3. Create pipeline with preprocessing and model
# 4. Define parameter grid
# 5. Perform GridSearchCV
# 6. Evaluate on test set
# 7. Analyze feature importance
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and fit
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### 3.1 K-Means
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Silhouette score
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.4f}")

# Elbow method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### 4.1 Cross-Validation
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# K-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 4.2 Grid Search
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
```

### 6.1 Complete Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Load data
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))

# Feature importance
importances = best_model.named_steps['classifier'].feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.4f}")
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Phase 2 Complete!
