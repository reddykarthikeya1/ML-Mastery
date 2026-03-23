# 6.3 Unsupervised Learning

## 🎯 Quick Overview
- **Unsupervised Learning**: Find patterns in unlabeled data
- **Clustering**: Group similar items
- **Dimensionality Reduction**: Compress data
- **Foundation for**: Data exploration, feature learning

---

## 1. Clustering

### 🧠 Mathematical Intuition & Logic Behind Algorithms

**1. K-Means Clustering**
*   **Logic:** Groups data into `K` distinct clusters based on feature similarity.
*   **Algorithm Steps:** 
    1. Randomly initialize `K` centroids.
    2. Assign each data point to the closest centroid (usually via **Euclidean distance**).
    3. Recalculate the centroids as the mean of all points assigned to that cluster.
    4. Repeat until centroids stop moving (convergence).
*   **Cost Function:** Minimizes **Inertia** (Within-Cluster Sum of Squares, WCSS).

**2. DBSCAN (Density-Based Spatial Clustering)**
*   **Logic:** Groups points that are closely packed together, marking points that lie alone in low-density regions as outliers.
*   **Math:** Relies on two parameters: `eps` (radius of neighborhood) and `min_samples` (minimum points to form a dense region). It doesn't require predefined `K` and perfectly isolates non-spherical clusters.

**3. Hierarchical Clustering (Agglomerative)**
*   **Logic:** Builds a hierarchy of clusters either bottom-up (Agglomerative) or top-down (Divisive).
*   **Math:** At each step, joins the two most similar clusters based on a Linkage metric (e.g., Ward's method minimizes the variance of merged clusters). Visualized via a **Dendrogram**.

### K-Means

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Train
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Evaluate
silhouette = silhouette_score(X, labels)
db_score = davies_bouldin_score(X, labels)

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# Elbow method for optimal k
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

# Cluster visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
           kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200)
plt.title('K-Means Clustering')
plt.show()
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch

# Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X)

# Dendrogram
linkage_matrix = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# Linkage methods
methods = ['ward', 'complete', 'average', 'single']

for method in methods:
    hc = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = hc.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"{method}: Silhouette = {score:.4f}")
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN

# DBSCAN (density-based)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Number of clusters (excluding noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title(f'DBSCAN (k={n_clusters})')
plt.show()
```

### Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture

# GMM
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X)

# Probabilities
proba = gmm.predict_proba(X)

# BIC/AIC for model selection
bic_scores = []
aic_scores = []

for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

plt.plot(range(1, 11), bic_scores, 'bo-', label='BIC')
plt.plot(range(1, 11), aic_scores, 'ro-', label='AIC')
plt.xlabel('k')
plt.legend()
plt.title('Model Selection')
plt.show()
```

---

## 2. Dimensionality Reduction

### 🧠 Mathematical Intuition & Logic Behind Algorithms

**1. PCA (Principal Component Analysis)**
*   **Logic:** Projects high-dimensional data onto a lower-dimensional subspace while preserving as much variance as possible.
*   **Math:** 
    1. Standardize the data (mean=0, variance=1).
    2. Compute the **Covariance Matrix** to understand feature relationships.
    3. Calculate the **Eigenvectors and Eigenvalues** of the covariance matrix.
    4. Sort Eigenvectors by highest Eigenvalue (these are your Principal Components -> directions of maximum variance).

**2. t-SNE & UMAP**
*   **Logic:** Non-linear dimensionality reduction primarily used for visualizing high-dimensional data in 2D/3D.
*   **Math (t-SNE):** Calculates similarity probabilities between points in high-dimensional space and low-dimensional space, minimizing the **Kullback-Leibler (KL) Divergence** between the two distributions.

### PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"PC1 explains: {explained_var[0]:.4f}")
print(f"PC2 explains: {explained_var[1]:.4f}")
print(f"Cumulative: {cumulative_var[0]+cumulative_var[1]:.4f}")

# Scree plot
plt.bar(range(1, len(explained_var)+1), explained_var)
plt.plot(range(1, len(explained_var)+1), cumulative_var, 'ro-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.show()

# Visualization
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.show()

# Optimal number of components
pca_full = PCA()
pca_full.fit(X_scaled)

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
optimal_k = np.argmax(cumulative_var >= 0.95) + 1

print(f"Components for 95% variance: {optimal_k}")
```

### t-SNE

```python
from sklearn.manifold import TSNE

# t-SNE (for visualization)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Projection')
plt.show()

# Different perplexity values
perplexities = [5, 10, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, perp in zip(axes.flatten(), perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X)
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1])
    ax.set_title(f'Perplexity = {perp}')

plt.tight_layout()
plt.show()
```

### UMAP

```python
import umap

# UMAP (faster than t-SNE)
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Projection')
plt.show()
```

---

## 3. Association Rule Learning

### Apriori Algorithm

```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Transaction data
data = {
    'Milk': [1, 1, 1, 1, 0, 0, 0, 0],
    'Bread': [1, 1, 0, 1, 1, 1, 0, 0],
    'Butter': [0, 1, 1, 1, 0, 1, 1, 0],
    'Eggs': [1, 0, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

---

## 4. Anomaly Detection

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
predictions = clf.fit_predict(X)

# -1 for outliers, 1 for inliers
n_outliers = sum(predictions == -1)
print(f"Number of outliers: {n_outliers}")

# Anomaly scores
scores = clf.decision_function(X)
```

### One-Class SVM

```python
from sklearn.svm import OneClassSVM

# One-Class SVM
ocsvm = OneClassSVM(nu=0.1, kernel='rbf')
predictions = ocsvm.fit_predict(X)

n_outliers = sum(predictions == -1)
print(f"Number of outliers: {n_outliers}")
```

---

## 💻 Python Code Examples

```python
# === Customer Segmentation ===

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate synthetic customer data
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
axes[0].set_title('Original Labels')

# K-Means
axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
axes[1].scatter(kmeans.cluster_centers_[:, 0], 
               kmeans.cluster_centers_[:, 1], 
               c='red', marker='x', s=200)
axes[1].set_title('K-Means Clustering')

# DBSCAN
axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
axes[2].set_title('DBSCAN Clustering')

plt.tight_layout()
plt.show()

# === PCA for Visualization ===

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(10, 8))
for class_label in np.unique(y):
    mask = y == class_label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               label=f'Class {class_label}')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# === Anomaly Detection ===

from sklearn.ensemble import IsolationForest
import numpy as np

# Generate data with outliers
np.random.seed(42)
X_normal = np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))
X = np.vstack([X_normal, X_outliers])

# Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
predictions = clf.fit_predict(X)

# Visualize
plt.figure(figsize=(10, 8))

# Inliers
inliers = X[predictions == 1]
plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', label='Inliers', alpha=0.6)

# Outliers
outliers = X[predictions == -1]
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', label='Outliers', marker='x')

plt.title('Anomaly Detection with Isolation Forest')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📊 Summary Tables

### Clustering Algorithms

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| K-Means | Fast, simple | Need k, spherical clusters | General purpose |
| Hierarchical | No k needed, dendrogram | Slow on large data | Small datasets |
| DBSCAN | No k needed, arbitrary shapes | Sensitive to eps | Density-based |
| GMM | Probabilistic, flexible | More parameters | Overlapping clusters |

### Dimensionality Reduction

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| PCA | Fast, interpretable | Linear | General purpose |
| t-SNE | Preserves local structure | Slow, stochastic | Visualization |
| UMAP | Fast, preserves structure | Newer, less tested | Visualization |

### Evaluation Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Silhouette | [-1, 1] | Higher is better |
| Davies-Bouldin | [0, ∞) | Lower is better |
| Inertia | [0, ∞) | Lower is better |
| Explained Variance | [0, 1] | Higher is better |

---

## 🎯 ML Applications

| Task | Algorithm | Industry |
|------|-----------|----------|
| Customer Segmentation | K-Means | Retail |
| Anomaly Detection | Isolation Forest | Finance |
| Feature Compression | PCA | All domains |
| Document Clustering | Hierarchical | NLP |
| Image Segmentation | DBSCAN | Computer Vision |

---

**Status:** ✅ Complete
**Next:** Model Evaluation
