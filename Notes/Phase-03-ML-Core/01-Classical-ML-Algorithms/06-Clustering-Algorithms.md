# 7.6 Clustering Algorithms

## 🎯 Learning Objectives
After completing this section, you will master:
1. **K-Means Clustering**: Algorithm, initialization methods, and choosing k
2. **DBSCAN**: Density-based clustering for arbitrary shapes
3. **Hierarchical Clustering**: Agglomerative clustering and dendrograms
4. **Gaussian Mixture Models**: Probabilistic clustering with EM algorithm
5. **Advanced Methods**: Spectral clustering, affinity propagation, and more

---

## 📚 Clustering Fundamentals

### What is Clustering?

**Definition:** Unsupervised learning technique that groups similar data points together

**Goal:** 
- **High intra-cluster similarity**: Points in same cluster are similar
- **Low inter-cluster similarity**: Points in different clusters are dissimilar

**Visual Representation:**
```
Before Clustering:          After Clustering:
    ●   ●                       ●   ●
  ●   ●   ●   ●               ●   ●   ●   ●
    ●   ●   ●                   ●   ●   ●
                                
    ○   ○                       ○   ○
  ○   ○   ○   ○               ○   ○   ○   ○
    ○   ○   ○                   ○   ○   ○

● = Cluster 1    ○ = Cluster 2
```

**Applications:**
- Customer segmentation
- Image segmentation
- Anomaly detection
- Document clustering
- Gene expression analysis

---

## 📚 K-Means Clustering

### 7.6.1 K-Means Algorithm

**Core Idea:** Partition data into k clusters where each point belongs to the nearest centroid

**Algorithm:**
```
1. Initialize k centroids (randomly or using K-Means++)
2. Repeat until convergence:
   a. Assignment step: Assign each point to nearest centroid
   b. Update step: Recalculate centroids as mean of assigned points
3. Return final centroids and assignments
```

**Visual Steps:**
```
Step 1: Initialize        Step 2: Assign         Step 3: Update
   ●   ●   ●                 ●   ●   ●              ●   ●   ●
 ●   ×   ●   ●             ●   ×   ●   ●          ●   ×   ●   ●
   ●   ●   ●   ○             ●   ●   ●   ○          ●   ●   ●   ○
     ×       ○                 ×       ○              ×       ○
   ○   ○   ○                 ○   ○   ○              ○   ○   ○

× = Centroid   ● = Cluster 1   ○ = Cluster 2
```

**Mathematical Formulation:**

**Objective Function (Inertia):**
$$J = \sum_{j=1}^{k} \sum_{i \in C_j} ||x_i - \mu_j||^2$$

Where:
- $k$: Number of clusters
- $C_j$: Set of points in cluster j
- $\mu_j$: Centroid of cluster j

**Goal:** Minimize J (within-cluster variance)

### K-Means Implementation from Scratch

```python
import numpy as np
from typing import Tuple, Optional

class KMeans:
    """
    K-Means clustering algorithm implemented from scratch.
    """
    
    def __init__(self, 
                 n_clusters: int = 8,
                 init: str = 'kmeans++',
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """
        Initialize K-Means.
        
        Args:
            n_clusters: Number of clusters (k)
            init: Initialization method ('random' or 'kmeans++')
            n_init: Number of times to run algorithm with different seeds
            max_iter: Maximum iterations per run
            tol: Tolerance for convergence
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.centroids = None
        self.labels = None
        self.inertia = None
    
    def _euclidean_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Calculate pairwise Euclidean distances"""
        # X: (n_samples, n_features)
        # centroids: (k, n_features)
        # Output: (n_samples, k)
        
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
        centroids_sq = np.sum(centroids ** 2, axis=1)  # (k,)
        cross_term = np.dot(X, centroids.T)  # (n, k)
        
        distances = X_sq + centroids_sq - 2 * cross_term
        distances = np.maximum(distances, 0)  # Numerical stability
        
        return np.sqrt(distances)
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using specified method"""
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Random selection
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[indices]
        
        elif self.init == 'kmeans++':
            # K-Means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            idx = np.random.randint(n_samples)
            centroids[0] = X[idx]
            
            # Choose remaining centroids
            for k in range(1, self.n_clusters):
                # Calculate distances to nearest centroid
                distances = self._euclidean_distance(X, centroids[:k])
                min_distances = np.min(distances, axis=1)
                
                # Square distances for probability
                squared_distances = min_distances ** 2
                probabilities = squared_distances / np.sum(squared_distances)
                
                # Choose next centroid with probability proportional to distance²
                idx = np.random.choice(n_samples, p=probabilities)
                centroids[k] = X[idx]
            
            return centroids
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid"""
        distances = self._euclidean_distance(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids as mean of assigned points"""
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                # Handle empty cluster - reinitialize randomly
                new_centroids[k] = X[np.random.randint(len(X))]
        
        return new_centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, 
                         centroids: np.ndarray) -> float:
        """Compute within-cluster sum of squares"""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                inertia += np.sum((X[mask] - centroids[k]) ** 2)
        return inertia
    
    def _fit_single(self, X: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Single run of K-Means"""
        np.random.seed(random_state)
        
        # Initialize centroids
        centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            if centroid_shift < self.tol:
                break
            
            centroids = new_centroids
        
        inertia = self._compute_inertia(X, labels, centroids)
        
        return centroids, labels, inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """Fit K-Means to data"""
        best_centroids = None
        best_labels = None
        best_inertia = float('inf')
        
        # Run multiple times with different initializations
        for i in range(self.n_init):
            random_state = self.random_state if self.random_state is not None else i
            centroids, labels, inertia = self._fit_single(X, random_state)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
        
        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        self.fit(X)
        return self.labels
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to cluster distance space"""
        return self._euclidean_distance(X, self.centroids)
```

### 7.6.2 Choosing k: Elbow Method and Silhouette Analysis

**Elbow Method:**
```python
def elbow_method(X, k_range, **kmeans_kwargs):
    """Find optimal k using elbow method"""
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        inertias.append(kmeans.inertia)
    
    return inertias


# Usage
import matplotlib.pyplot as plt

k_range = range(1, 11)
inertias = elbow_method(X, k_range)

plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Look for the "elbow" point where inertia starts decreasing linearly
```

**Silhouette Analysis:**
```python
def silhouette_score(X, labels):
    """Calculate silhouette score"""
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters >= n_samples:
        return 0.0
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Calculate a(i): mean distance to points in same cluster
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude self
        
        if np.sum(same_cluster) > 0:
            a_i = np.mean(np.sqrt(np.sum((X[same_cluster] - X[i]) ** 2, axis=1)))
        else:
            a_i = 0
        
        # Calculate b(i): min mean distance to points in other clusters
        b_i = float('inf')
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = labels == label
                if np.sum(other_cluster) > 0:
                    mean_dist = np.mean(np.sqrt(np.sum((X[other_cluster] - X[i]) ** 2, axis=1)))
                    b_i = min(b_i, mean_dist)
        
        if b_i == float('inf'):
            b_i = 0
        
        # Silhouette coefficient
        if max(a_i, b_i) > 0:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_vals[i] = 0
    
    return np.mean(silhouette_vals)


# Find optimal k using silhouette score
from sklearn.metrics import silhouette_score as sklearn_silhouette

k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = sklearn_silhouette(X, labels)
    silhouette_scores.append(score)

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k: {optimal_k} (silhouette score: {max(silhouette_scores):.4f})")
```

### K-Means Variants

**K-Means++:**
- Better initialization than random
- Reduces chance of poor local optima
- Default in most implementations

**Mini-Batch K-Means:**
```python
class MiniBatchKMeans:
    """
    Mini-batch K-Means for large datasets.
    Faster but slightly less accurate than full K-Means.
    """
    
    def __init__(self, n_clusters=8, batch_size=100, max_iter=100):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.centroids = None
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()
        
        # Running counts for each centroid
        counts = np.ones(self.n_clusters)
        
        for iteration in range(self.max_iter):
            # Sample mini-batch
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[batch_indices]
            
            # Assign to nearest centroid
            distances = np.sqrt(((X_batch[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            
            # Update centroids with learning rate
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    batch_centroid = np.mean(X_batch[mask], axis=0)
                    learning_rate = 1.0 / counts[k]
                    self.centroids[k] = (1 - learning_rate) * self.centroids[k] + learning_rate * batch_centroid
                    counts[k] += np.sum(mask)
        
        return self
```

---

## 📚 DBSCAN (Density-Based Spatial Clustering)

### 7.6.3 DBSCAN Fundamentals

**Core Idea:** Cluster points in dense regions, mark sparse regions as noise

**Key Concepts:**
```
ε (eps): Neighborhood radius
min_samples: Minimum points to form dense region

Point Types:
- Core Point: Has ≥ min_samples points within ε
- Border Point: Has < min_samples points within ε but is in neighborhood of a core point
- Noise Point: Neither core nor border point
```

**Visual:**
```
    ● ● ●           ○ = Core point (≥ min_samples in ε)
  ● ● ○ ● ●         ● = Border point
    ● ● ●           × = Noise point
        ×
    ×       ● ●
            ● ○ ●
            ● ● ●
```

**DBSCAN Algorithm:**
```
1. Mark all points as unvisited
2. For each unvisited point p:
   a. Mark p as visited
   b. Find all points within ε of p (neighbors)
   c. If neighbors < min_samples:
      - Mark p as noise
   d. Else:
      - Create new cluster C
      - Add p to C
      - For each neighbor n:
        * If n is unvisited, mark as visited and check its neighbors
        * If n is not in any cluster, add to C
        * If n is a core point, add its neighbors to process
3. Return clusters
```

### DBSCAN Implementation

```python
class DBSCAN:
    """
    DBSCAN clustering algorithm implemented from scratch.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean'):
        """
        Initialize DBSCAN.
        
        Args:
            eps: Maximum distance between two points to be considered neighbors
            min_samples: Minimum points to form a dense region
            metric: Distance metric
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels = None
        self.core_sample_indices = None
    
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix"""
        n_samples = len(X)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _get_neighbors(self, distance_matrix: np.ndarray, point_idx: int) -> np.ndarray:
        """Get all points within eps of given point"""
        return np.where(distance_matrix[point_idx] <= self.eps)[0]
    
    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """Fit DBSCAN to data"""
        n_samples = len(X)
        
        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(X)
        
        # Find core points
        core_sample_indices = []
        neighbors_list = []
        
        for i in range(n_samples):
            neighbors = self._get_neighbors(distance_matrix, i)
            neighbors_list.append(neighbors)
            
            if len(neighbors) >= self.min_samples:
                core_sample_indices.append(i)
        
        self.core_sample_indices = np.array(core_sample_indices)
        
        # Initialize labels (-1 = noise, -2 = unvisited)
        labels = np.full(n_samples, -2, dtype=int)
        cluster_id = -1
        
        # Process each point
        for i in range(n_samples):
            if labels[i] != -2:
                continue  # Already processed
            
            neighbors = neighbors_list[i]
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Mark as noise
            else:
                # Start new cluster
                cluster_id += 1
                labels[i] = cluster_id
                
                # Expand cluster
                seed_set = set(neighbors) - {i}
                
                while seed_set:
                    current_point = seed_set.pop()
                    
                    if labels[current_point] == -1:
                        labels[current_point] = cluster_id  # Change noise to border point
                    elif labels[current_point] == -2:
                        labels[current_point] = cluster_id
                        
                        # If core point, add its neighbors
                        current_neighbors = neighbors_list[current_point]
                        if len(current_neighbors) >= self.min_samples:
                            seed_set |= set(current_neighbors) - {current_point}
        
        self.labels = labels
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        self.fit(X)
        return self.labels
    
    def get_noise_samples(self) -> np.ndarray:
        """Get indices of noise points"""
        return np.where(self.labels == -1)[0]
```

### DBSCAN vs K-Means

| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| Cluster Shape | Spherical | Arbitrary |
| Number of Clusters | Must specify k | Automatically determined |
| Noise Handling | Sensitive | Robust (explicitly handles noise) |
| Parameters | k | eps, min_samples |
| Complexity | O(n·k·i) | O(n²) or O(n log n) with indexing |
| Density Variation | Poor | Good |

---

## 📚 Hierarchical Clustering

### 7.6.4 Agglomerative Clustering

**Core Idea:** Build hierarchy of clusters bottom-up

**Algorithm:**
```
1. Start with each point as its own cluster
2. Repeat until one cluster remains:
   a. Find two closest clusters
   b. Merge them
3. Cut dendrogram at desired level for final clusters
```

**Visual (Dendrogram):**
```
Height (Distance)
  ↑
  |           ┌───┴───┐
  |       ┌───┤       │
  |   ┌───┤   │   ┌───┴───┐
  |   │   │   │   │       │
  | ──┴─┬─┴─┬─┴─┬─┴───┬───┴──
  |     │   │   │     │
  |     A   B   C     D    E  ← Data points
  
Cut at height h → 3 clusters: {A,B}, {C}, {D,E}
```

### Linkage Criteria

**1. Single Linkage:**
```
distance(A, B) = min(dist(a, b) for a in A, b in B)
- Nearest neighbor
- Can cause "chaining" effect
```

**2. Complete Linkage:**
```
distance(A, B) = max(dist(a, b) for a in A, b in B)
- Farthest neighbor
- Produces compact clusters
```

**3. Average Linkage:**
```
distance(A, B) = mean(dist(a, b) for a in A, b in B)
- Balanced approach
- Most commonly used
```

**4. Ward's Method:**
```
distance(A, B) = increase in total within-cluster variance
- Minimizes variance
- Similar to K-Means objective
```

### Hierarchical Clustering Implementation

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

class AgglomerativeClustering:
    """
    Agglomerative hierarchical clustering.
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'ward'):
        """
        Initialize Agglomerative Clustering.
        
        Args:
            n_clusters: Number of clusters to form
            linkage: Linkage criterion ('single', 'complete', 'average', 'ward')
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage
        self.labels = None
        self.children = None
        self.distances = None
    
    def fit(self, X: np.ndarray) -> 'AgglomerativeClustering':
        """Fit hierarchical clustering"""
        n_samples = len(X)
        
        # Compute pairwise distances
        distance_matrix = pdist(X, metric='euclidean')
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method=self.linkage_method)
        
        self.children = linkage_matrix[:, :2].astype(int)
        self.distances = linkage_matrix[:, 2]
        
        # Assign cluster labels
        from scipy.cluster.hierarchy import fcluster
        self.labels = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust')
        self.labels -= 1  # Convert to 0-indexed
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict"""
        self.fit(X)
        return self.labels
    
    def plot_dendrogram(self, **kwargs):
        """Plot dendrogram"""
        plt.figure(figsize=(10, 6))
        dendrogram(
            np.column_stack([self.children, self.distances]),
            **kwargs
        )
        plt.title('Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    # Generate data
    X, _ = make_blobs(n_samples=50, centers=4, random_state=42)
    
    # Fit hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels = hc.fit_predict(X)
    
    # Plot dendrogram
    hc.plot_dendrogram()
```

---

## 📚 Gaussian Mixture Models (GMM)

### 7.6.5 GMM Fundamentals

**Core Idea:** Data is generated from a mixture of Gaussian distributions

**Model:**
$$P(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x | \mu_k, \Sigma_k)$$

Where:
- $\pi_k$: Mixing coefficient (weight) for component k, $\sum \pi_k = 1$
- $\mathcal{N}(x | \mu_k, \Sigma_k)$: Gaussian distribution with mean $\mu_k$ and covariance $\Sigma_k$

**Gaussian Distribution:**
$$\mathcal{N}(x | \mu, \Sigma) = \frac{1}{\sqrt{2\pi|\Sigma|}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$

### 7.6.6 Expectation-Maximization (EM) Algorithm

**Problem:** Find parameters ($\pi_k, \mu_k, \Sigma_k$) that maximize likelihood

**EM Algorithm:**
```
1. Initialize parameters (π, μ, Σ)
2. Repeat until convergence:
   
   E-step (Expectation):
   γ(z_nk) = π_k · N(x_n | μ_k, Σ_k) / Σ_j π_j · N(x_n | μ_j, Σ_j)
   
   M-step (Maximization):
   N_k = Σ_n γ(z_nk)
   π_k = N_k / N
   μ_k = (1/N_k) · Σ_n γ(z_nk) · x_n
   Σ_k = (1/N_k) · Σ_n γ(z_nk) · (x_n - μ_k)(x_n - μ_k)^T
   
3. Return final parameters
```

**Visual:**
```
Initial:                 After EM:
   ● ● ●                    ● ● ●
 ● ●   ● ●                ● ●   ● ●
   ● ● ●   ○ ○              ● ● ●   ○ ○
     ×       ○                ×       ○
   ○ ○ ○   ○ ○              ○ ○ ○   ○ ○
     ○ ○   ○                  ○ ○   ○

× = Initial μ           × = Final μ
Shaded = Soft assignment
```

### GMM Implementation

```python
class GaussianMixtureModel:
    """
    Gaussian Mixture Model with EM algorithm.
    """
    
    def __init__(self, 
                 n_components: int = 2,
                 covariance_type: str = 'full',
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 random_state: Optional[int] = None):
        """
        Initialize GMM.
        
        Args:
            n_components: Number of mixture components (K)
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            random_state: Random seed
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.weights = None  # π_k
        self.means = None    # μ_k
        self.covariances = None  # Σ_k
        self.converged = False
    
    def _multivariate_gaussian(self, X: np.ndarray, mean: np.ndarray, 
                                cov: np.ndarray) -> np.ndarray:
        """Calculate multivariate Gaussian probability"""
        n_features = X.shape[1]
        
        if self.covariance_type == 'spherical':
            # Spherical covariance (single variance)
            var = cov[0, 0]
            norm_coeff = 1.0 / np.sqrt((2 * np.pi) ** n_features * var ** n_features)
            exponent = -np.sum((X - mean) ** 2, axis=1) / (2 * var)
            return norm_coeff * np.exp(exponent)
        
        elif self.covariance_type == 'diag':
            # Diagonal covariance
            var = np.diag(cov)
            norm_coeff = 1.0 / np.sqrt((2 * np.pi) ** n_features * np.prod(var))
            exponent = -np.sum((X - mean) ** 2 / var, axis=1) / 2
            return norm_coeff * np.exp(exponent)
        
        else:  # 'full' or 'tied'
            # Full covariance
            diff = X - mean
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
            
            norm_coeff = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            
            return norm_coeff * np.exp(exponent)
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """E-step: Calculate responsibilities"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k]
            )
        
        # Normalize
        total = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities /= (total + 1e-10)
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """M-step: Update parameters"""
        n_samples, n_features = X.shape
        
        # Effective number of points per component
        N_k = np.sum(responsibilities, axis=0)
        
        # Update weights
        self.weights = N_k / n_samples
        
        # Update means
        self.means = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            self.means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
        
        # Update covariances
        self.covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            
            if self.covariance_type == 'full':
                self.covariances[k] = (weighted_diff.T @ diff) / N_k[k]
            elif self.covariance_type == 'tied':
                self.covariances += (weighted_diff.T @ diff) / n_samples
            elif self.covariance_type == 'diag':
                self.covariances[k] = np.diag(
                    np.sum(weighted_diff * diff, axis=0) / N_k[k]
                )
            elif self.covariance_type == 'spherical':
                var = np.sum(weighted_diff * diff) / (N_k[k] * n_features)
                self.covariances[k] = var * np.eye(n_features)
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood of data"""
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)
        
        for k in range(self.n_components):
            likelihood += self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k]
            )
        
        return np.sum(np.log(likelihood + 1e-10))
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureModel':
        """Fit GMM using EM algorithm"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        
        if self.covariance_type == 'spherical':
            self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        else:
            self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Check convergence
            log_likelihood = self._compute_log_likelihood(X)
            
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged = True
                break
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster probabilities (responsibilities)"""
        return self._e_step(X)
    
    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """Compute log-likelihood (BIC/AIC can also be computed)"""
        return self._compute_log_likelihood(X)
```

---

## 📊 Summary Tables

### Clustering Algorithm Comparison

| Algorithm | Type | Parameters | Pros | Cons |
|-----------|------|------------|------|------|
| **K-Means** | Centroid-based | k | Fast, simple | Spherical clusters, needs k |
| **DBSCAN** | Density-based | eps, min_samples | Arbitrary shapes, handles noise | Sensitive to parameters |
| **Hierarchical** | Hierarchical | n_clusters, linkage | Dendrogram, no need for k | O(n²) complexity |
| **GMM** | Distribution-based | n_components | Soft clustering, probabilistic | Can overfit, needs initialization |

### Choosing Number of Clusters

| Method | Metric | How to Choose |
|--------|--------|---------------|
| **Elbow** | Inertia | Look for "elbow" in plot |
| **Silhouette** | Silhouette score | Maximize score (-1 to 1) |
| **Gap Statistic** | Gap statistic | Maximize gap |
| **BIC/AIC** | Information criteria | Minimize BIC/AIC (for GMM) |

### Clustering Validation Metrics

| Metric | Formula | Range | Best |
|--------|---------|-------|------|
| **Inertia** | Σ\|\|x - μ\|\|² | [0, ∞) | Lower |
| **Silhouette** | (b-a)/max(a,b) | [-1, 1] | Higher |
| **Davies-Bouldin** | avg(max(R_ij)) | [0, ∞) | Lower |
| **Calinski-Harabasz** | Between/Within | [0, ∞) | Higher |

---

## 🎯 ML Applications

| Application | Algorithm | Description |
|-------------|-----------|-------------|
| **Customer Segmentation** | K-Means, GMM | Group customers by behavior |
| **Image Segmentation** | K-Means, DBSCAN | Partition image into regions |
| **Anomaly Detection** | DBSCAN, GMM | Identify outliers |
| **Document Clustering** | K-Means, Hierarchical | Group similar documents |
| **Gene Expression** | Hierarchical, GMM | Find gene patterns |
| **Social Network Analysis** | DBSCAN, Spectral | Detect communities |

---

## 📝 Practice Problems

### Level 1: Basic

1. **Conceptual**: Explain the difference between K-Means and DBSCAN
2. **Calculation**: Given 3 points and 2 centroids, calculate assignments and new centroids
3. **Understanding**: What is the "elbow method" and when does it fail?
4. **Code**: Implement silhouette score calculation
5. **Analysis**: Why does K-Means struggle with non-spherical clusters?

### Level 2: Intermediate

1. **Implementation**: Build complete K-Means with K-Means++ initialization
2. **Experiment**: Compare K-Means, DBSCAN, and GMM on the same dataset
3. **Analysis**: Investigate how eps and min_samples affect DBSCAN results
4. **Application**: Build customer segmentation using K-Means on mall data
5. **Visualization**: Create dendrograms and interpret cluster hierarchies

### Level 3: Advanced

1. **Research**: Implement spectral clustering from scratch
2. **Optimization**: Add parallel processing to K-Means for large datasets
3. **Extension**: Implement GMM with different covariance types
4. **Project**: Build anomaly detection system using DBSCAN and GMM
5. **Analysis**: Compare soft clustering (GMM) vs hard clustering (K-Means)

---

## 🔗 Related Topics
- [[03-Instance-Based-Learning]] - KNN distance metrics
- [[07-Dimensionality-Reduction]] - PCA before clustering
- [[08-Regularization-Techniques]] - Preventing overfitting in GMM
- [[Phase-04-Deep-Learning]] - Deep clustering methods

---

**Status:** ✅ Complete  
**Next:** [[01-Linear-Models]] (Phase 3 review) or [[02-Deep-Learning-Fundamentals/03-Regularization-Techniques]]
