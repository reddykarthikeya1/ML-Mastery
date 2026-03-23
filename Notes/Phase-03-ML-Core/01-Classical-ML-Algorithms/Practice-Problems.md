# Classical ML Algorithms - Practice Problems

## Topic 1: Linear Models

### Level 1: Basic

**1.1** Implement Ridge Regression:
```python
from sklearn.linear_model import Ridge

# 1. Create Ridge model with alpha=1.0
# 2. Fit on training data
# 3. Evaluate with cross-validation
```

**1.2** Lasso Feature Selection:
```python
from sklearn.linear_model import Lasso

# 1. Fit Lasso with alpha=0.1
# 2. Count non-zero coefficients
# 3. Print selected features
```

---

## Topic 2: Tree-Based Models

### Level 1: Basic

**2.1** Decision Tree:
```python
from sklearn.tree import DecisionTreeClassifier

# 1. Create tree with max_depth=3
# 2. Fit on data
# 3. Get feature importances
# 4. Visualize tree
```

**2.2** Random Forest:
```python
from sklearn.ensemble import RandomForestClassifier

# 1. Create RF with 100 trees
# 2. Fit on data
# 3. Evaluate accuracy
# 4. Get feature importances
```

### Level 2: Intermediate

**2.3** Compare Ensemble Methods:
```python
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier
)

# 1. Train all three models
# 2. Compare accuracy
# 3. Compare training time
# 4. Cross-validate best model
```

**2.4** XGBoost Tuning:
```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# 1. Define parameter grid for XGBoost
# 2. Perform GridSearchCV
# 3. Get best parameters
# 4. Evaluate on test set
```

---

## Topic 3: Instance-Based, SVM, Naive Bayes

### Level 1: Basic

**3.1** KNN Optimal k:
```python
from sklearn.neighbors import KNeighborsClassifier

# 1. Test k from 1 to 30
# 2. Plot CV accuracy vs k
# 3. Find optimal k
```

**3.2** SVM Kernel Comparison:
```python
from sklearn.svm import SVC

# 1. Test linear, poly, rbf, sigmoid kernels
# 2. Compare accuracy
# 3. Choose best kernel
```

### Level 2: Intermediate

**3.3** Naive Bayes Variants:
```python
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, BernoulliNB
)

# 1. Train all three variants
# 2. Compare performance
# 3. Identify best use case for each
```

---

## Topic 4: Clustering

### Level 1: Basic

**4.1** K-Means Elbow Method:
```python
from sklearn.cluster import KMeans

# 1. Test k from 1 to 10
# 2. Plot inertias
# 3. Find optimal k using elbow
```

**4.2** DBSCAN:
```python
from sklearn.cluster import DBSCAN

# 1. Fit DBSCAN with eps=0.5, min_samples=5
# 2. Count clusters and noise points
# 3. Visualize clusters
```

### Level 2: Intermediate

**4.3** Compare Clustering Algorithms:
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Train all three algorithms
# 2. Compare silhouette scores
# 3. Visualize results
# 4. Discuss pros/cons of each
```

---

## Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1 Ridge Regression
```python
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
scores = cross_val_score(ridge, X_train, y_train, cv=5)
```

### 2.1 Decision Tree
```python
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
importances = tree.feature_importances_
plot_tree(tree, feature_names=feature_names)
```

### 2.4 XGBoost Tuning
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
```

### 3.1 KNN Optimal k
```python
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_range[np.argmax(cv_scores)]
```

### 4.3 Compare Clustering
```python
models = {
    'K-Means': KMeans(n_clusters=3),
    'DBSCAN': DBSCAN(eps=0.5),
    'Hierarchical': AgglomerativeClustering(n_clusters=3)
}

for name, model in models.items():
    labels = model.fit_predict(X)
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"{name} Silhouette: {score:.4f}")
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Classical ML Algorithms Complete!
