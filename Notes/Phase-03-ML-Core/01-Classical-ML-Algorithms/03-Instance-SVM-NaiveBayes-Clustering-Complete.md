# 7.3-7.6 Classical ML Algorithms

## 🎯 Quick Overview
- **Instance-Based**: KNN, LVQ
- **SVM**: Maximum margin classifier
- **Naive Bayes**: Probabilistic classifier
- **Clustering**: K-Means, DBSCAN, Hierarchical
- **Foundation for**: Diverse ML applications

---

## 1. Instance-Based Learning

### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
accuracy = knn.score(X_test, y_test)
print(f"KNN Accuracy: {accuracy:.4f}")

# Choose optimal k
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

plt.plot(k_range, cv_scores, 'bo-')
plt.xlabel('k')
plt.ylabel('CV Accuracy')
plt.title('Optimal k for KNN')
plt.show()
```

### Distance Metrics

```python
from sklearn.neighbors import KNeighborsClassifier

# Different distance metrics
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)

for name, model in [('Euclidean', knn_euclidean), 
                     ('Manhattan', knn_manhattan),
                     ('Minkowski', knn_minkowski)]:
    model.fit(X_train, y_train)
    print(f"{name} Accuracy: {model.score(X_test, y_test):.4f}")
```

---

## 2. Support Vector Machines (SVM)

### Binary Classification

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=100, n_features=2, 
                           n_informative=2, n_redundant=0,
                           random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Evaluate
accuracy = svm.score(X_test, y_test)
print(f"SVM Accuracy: {accuracy:.4f}")
```

### Kernel Functions

```python
from sklearn.svm import SVC

# Different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train, y_train)
    print(f"{kernel} Accuracy: {svm.score(X_test, y_test):.4f}")
```

### SVM for Regression (SVR)

```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Generate regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVR
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_train, y_train)

# Predict
y_pred = svr.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"SVR RMSE: {rmse:.4f}")
```

---

## 3. Naive Bayes

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gaussian NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)

# Evaluate
accuracy = gnb.score(X_test, y_test)
print(f"Gaussian NB Accuracy: {accuracy:.4f}")
```

### Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

# For count data (e.g., text)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

print(f"Multinomial NB Accuracy: {mnb.score(X_test, y_test):.4f}")
```

### Bernoulli Naive Bayes

```python
from sklearn.naive_bayes import BernoulliNB

# For binary features
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

print(f"Bernoulli NB Accuracy: {bnb.score(X_test, y_test):.4f}")
```

---

## 4. Clustering Algorithms

### K-Means

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Evaluate
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette:.4f}")

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

### DBSCAN

```python
from sklearn.cluster import DBSCAN

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters: {n_clusters}")
print(f"Noise points: {n_noise}")
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X)

# Dendrogram
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.show()
```

---

## 💻 Python Code Examples

```python
# === Complete Classical ML Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = model.score(X_test_scaled, y_test)
    
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Cross-Validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 📊 Summary Tables

### Algorithm Comparison

| Algorithm | Type | Speed | Accuracy | Use Case |
|-----------|------|-------|----------|----------|
| KNN | Instance-based | Slow prediction | Medium | Small datasets |
| SVM | Margin-based | Medium | High | High-dimensional |
| Naive Bayes | Probabilistic | Fast | Medium | Text classification |
| Random Forest | Ensemble | Medium | High | General purpose |

### Clustering Algorithms

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| K-Means | Fast, simple | Need k, spherical | General purpose |
| DBSCAN | No k needed, arbitrary shapes | Sensitive to eps | Density-based |
| Hierarchical | No k needed, dendrogram | Slow | Small datasets |

---

## 🎯 ML Applications

| Algorithm | ML Application |
|-----------|----------------|
| KNN | Recommendation systems |
| SVM | Text classification, Image classification |
| Naive Bayes | Spam filtering, Sentiment analysis |
| K-Means | Customer segmentation |
| DBSCAN | Anomaly detection |

---

**Status:** ✅ Complete
**Next:** Deep Learning Fundamentals
