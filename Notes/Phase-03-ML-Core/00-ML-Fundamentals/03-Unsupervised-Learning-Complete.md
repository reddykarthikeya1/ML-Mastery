# 6.3 Unsupervised Learning

## 🎯 Quick Overview
- **Unsupervised Learning**: Find patterns in unlabeled data
- **Clustering**: Group similar items
- **Dimensionality Reduction**: Compress data
- **Foundation for**: Data exploration, feature learning

---

## 1. Clustering

### K-Means

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Train
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

---

## 2. Dimensionality Reduction

### PCA

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.4f}")

# Scree plot
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), 
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.show()
```

### t-SNE

```python
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title('t-SNE Projection')
plt.show()
```

---

## 💻 Python Code Examples

```python
# === Complete Unsupervised Learning Pipeline ===

from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# === Clustering Example ===

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, 
                        cluster_std=0.60, random_state=0)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans_labels = kmeans.fit_predict(X)

# Evaluate
kmeans_silhouette = silhouette_score(X, kmeans_labels)
kmeans_db = davies_bouldin_score(X, kmeans_labels)

print(f"K-Means Silhouette: {kmeans_silhouette:.4f}")
print(f"K-Means DB Score: {kmeans_db:.4f}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
axes[0].set_title('True Labels')

axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
axes[1].set_title('K-Means')

axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
axes[2].set_title('DBSCAN')

plt.tight_layout()
plt.show()

# === Dimensionality Reduction Example ===

# Load iris dataset
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
```

---

## 📊 Summary Tables

### Clustering Algorithms

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| K-Means | Fast, simple | Need k, spherical clusters | General purpose |
| Hierarchical | No k needed, dendrogram | Slow on large data | Small datasets |
| DBSCAN | No k needed, arbitrary shapes | Sensitive to eps | Density-based |

### Dimensionality Reduction

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| PCA | Fast, interpretable | Linear | General purpose |
| t-SNE | Preserves local structure | Slow, stochastic | Visualization |
| UMAP | Fast, preserves structure | Newer | Visualization |

---

## 🎯 ML Applications

| Task | Algorithm | Industry |
|------|-----------|----------|
| Customer Segmentation | K-Means | Retail |
| Anomaly Detection | DBSCAN | Finance |
| Feature Compression | PCA | All domains |
| Document Clustering | Hierarchical | NLP |

---

---

## ❓ Quick Check Questions

1. What is the goal of K-Means clustering?
2. How does DBSCAN differ from K-Means regarding the number of clusters?
3. What is a Dendrogram?
4. What is the purpose of Principal Component Analysis (PCA)?
5. When should you use t-SNE instead of PCA?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **K-Means clustering** aims to partition $n$ observations into $K$ clusters in which each observation belongs to the cluster with the nearest mean (centroid).
2. **K-Means** requires you to specify the number of clusters ($K$) in advance. **DBSCAN** automatically determines the number of clusters based on the density of the data points and can identify outliers as noise.
3. A **Dendrogram** is a tree-like diagram that records the sequences of merges or splits in hierarchical clustering.
4. **PCA** is a dimensionality reduction technique that transforms a large set of variables into a smaller one that still contains most of the information (variance) from the original set.
5. Use **t-SNE** when your primary goal is the **visualization** of high-dimensional data in 2D or 3D, as it is much better at preserving local structures and revealing clusters than PCA.

</details>

---

**Status:** ✅ Complete
**Next:** Model Evaluation
