# 6.3 Unsupervised Learning

## 🎯 Quick Overview
- **Unsupervised Learning**: Find patterns in unlabeled data
- **Clustering**: Group similar items
- **Dimensionality Reduction**: Compress data
- **Foundation for**: Data exploration, feature learning

---

#### 🧒 ELI5: Unsupervised Learning, Clustering & PCA

> Imagine you're organizing a messy room full of toys with NO labels.
>
> **Unsupervised Learning** (No teacher, no answers):
> - Nobody tells you "these are cars, these are dolls"
> - You have to FIGURE OUT the patterns yourself
> - "Hmm, these all have wheels... these all have hair..."
>
> **Clustering** (Grouping similar things):
>
> **K-Means** (Pre-decide number of groups):
> - You say: "I want 3 toy boxes"
> - Algorithm: "OK, I'll find the BEST 3 groups"
> - Groups: Cars, Dolls, Blocks
> - **Elbow Method**: How do you know 3 is right?
>   - Try 1 group: All toys in one box (bad!)
>   - Try 2 groups: Better but still mixed
>   - Try 3 groups: Much better! ← "Elbow" point
>   - Try 10 groups: Each toy in own box (overkill!)
>
> **Hierarchical Clustering** (Family tree of toys):
> - Start: Every toy is its own group (100 groups)
> - Merge closest two: "Car #1 and Car #2 are similar"
> - Keep merging until all in one big group
> - **Dendrogram**: Tree diagram showing merge history
> - Cut the tree at any height = any number of groups!
>
> **DBSCAN** (Find groups without pre-deciding):
> - "Toys within 1 foot of each other = same group"
> - Finds ANY shape clusters (not just circles like K-Means)
> - "This toy is alone in the corner? It's NOISE (outlier)"
>
> **PCA - Dimensionality Reduction** (Compressing data):
>
> Imagine describing a toy car:
> - **Original**: Length, width, height, weight, color, material, wheels, windows... (10 features!)
> - **After PCA**: 
>   - PC1 = "Size" (combines length, width, height, weight)
>   - PC2 = "Appearance" (combines color, material)
>   - Now only 2 features instead of 10!
> - Like making a summary: "It's a big red car" instead of 10 separate facts
>
> **Why PCA?**:
> - Faster training (fewer features)
> - Remove redundant info (length & width often go together)
> - Visualize high-D data in 2D/3D
> - **Explained Variance**: "PC1 captures 60% of what makes toys different"

</details>

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

#### 🧒 ELI5: Advanced Clustering (GMM, DBSCAN) & Dimensionality Reduction (t-SNE, UMAP)

> Imagine you're organizing a music festival.
>
> **K-Means Limitation** (Only finds circular crowds):
> - Assumes clusters are CIRCULAR
> - What if fans are in a DONUT shape?
> - K-Means: "I see two circles" (wrong!)
> - Real: One donut-shaped crowd!
>
> **DBSCAN** (Density-based - finds ANY shape):
>
> **How it works**:
> - "People within 10 feet of each other = same group"
> - eps=10 (how close to be neighbors)
> - min_samples=5 (need 5 friends to form a group)
>
> **DBSCAN finds**:
> - Dense areas = clusters
> - Sparse areas = noise/outliers
> - ANY shape: circles, donuts, snakes, stars!
>
> **Like**: "If you and 4 friends are close, you're a group!"
>
> **Advantages over K-Means**:
> - ✅ Finds any shape
> - ✅ Doesn't need number of clusters
> - ✅ Finds outliers ("This person has no friends nearby")
> - ❌ Struggles with varying densities
>
> **Gaussian Mixture Models (GMM)** (Soft clustering):
>
> **K-Means** (Hard assignment):
> - "You're EITHER in cluster A OR cluster B"
> - Like: "You're EITHER a cat person OR a dog person"
>
> **GMM** (Soft assignment):
> - "You're 70% cluster A, 30% cluster B"
> - Like: "You like BOTH cats and dogs!"
>
> **How GMM works**:
> - Each cluster is a "bell curve" (Gaussian distribution)
> - Data point: "I'm somewhat close to center A, somewhat close to center B"
> - Probability: "60% likely from A, 40% likely from B"
>
> **When to use GMM**:
> - Overlapping clusters
> - Need probabilities (not just labels)
> - Elliptical clusters (stretched circles)
>
> **t-SNE** (Visualizing high-dimensional data):
>
> **Problem**: Your data has 100 features!
> - Can't visualize 100 dimensions!
> - How to show on 2D paper?
>
> **t-SNE Solution**:
> - "Similar things stay close, different things go far"
> - High-D: Point A and B are similar
> - Low-D (2D): A and B are neighbors!
> - High-D: Point C is very different
> - Low-D (2D): C is on the other side!
>
> **Like**: Making a seating chart for a party
> - Friends sit together
> - Enemies sit apart
> - 100-person party → fits on 2D room!
>
> **t-SNE vs PCA**:
> - **PCA**: Linear (straight lines), preserves GLOBAL structure
> - **t-SNE**: Non-linear (curvy), preserves LOCAL structure
> - PCA: "These 2 points are far" (accurate distance)
> - t-SNE: "These 2 points are neighbors" (accurate neighborhoods)
>
> **UMAP** (Newer, faster t-SNE):
>
> **UMAP** = "t-SNE but better"
> - ✅ Faster (seconds vs minutes)
> - ✅ Preserves BOTH local and global structure
> - ✅ Works on larger datasets
> - ✅ Can transform NEW data (t-SNE can't!)
>
> **Like**: "t-SNE's smarter cousin"
>
> **When to use which**:
> - **PCA**: Quick visualization, linear patterns, need speed
> - **t-SNE**: Beautiful plots, local patterns, publication figures
> - **UMAP**: Best of both worlds (recommended!)

</details>

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
