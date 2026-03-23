# 7.3 Instance-Based Learning

## 🎯 Learning Objectives
After completing this section, you will master:
1. **KNN Algorithm**: Understand how K-Nearest Neighbors works for classification and regression
2. **Distance Metrics**: Master Euclidean, Manhattan, Minkowski, and other distance functions
3. **KD-Trees and Ball Trees**: Implement efficient search structures for large datasets
4. **Learning Vector Quantization**: Understand prototype-based learning approaches
5. **Practical Implementation**: Build KNN and LVQ from scratch and apply to real problems

---

## 📚 Instance-Based Learning Fundamentals

### What is Instance-Based Learning?

**Instance-based learning** (also called **memory-based learning** or **lazy learning**) is a family of algorithms that:
- **Store training instances** in memory
- **Defer computation** until prediction time
- **Make predictions** based on similarity to stored instances

**Key Characteristics:**
```
Eager Learning (e.g., Decision Trees, Neural Networks)
├── Training: Build explicit model (slow)
└── Prediction: Apply model (fast)

Lazy Learning (e.g., KNN, Case-Based Reasoning)
├── Training: Store instances (fast)
└── Prediction: Compute from instances (slow)
```

**When to Use Instance-Based Learning:**
- Non-linear decision boundaries
- Multi-modal distributions
- Simple, interpretable baseline
- Small to medium-sized datasets

---

## 📚 K-Nearest Neighbors (KNN)

### 7.3.1 KNN Algorithm

**Core Idea:** "Tell me who your neighbors are, and I'll tell you who you are"

**Algorithm Steps:**
```
1. Store all training examples (X_train, y_train)
2. For a new test point x_test:
   a. Calculate distance to all training points
   b. Find k nearest neighbors
   c. For classification: majority vote
   d. For regression: average of neighbors
3. Return prediction
```

**Visual Representation:**
```
         Test Point (?)
              ●
             /|\
            / | \
           /  |  \
      k=3 /   |   \ k=5
         /    |    \
        ●     ●     ●
       K1    K2    K3
      (A)   (A)   (B)
      
      Prediction: Class A (2 vs 1)
```

**Mathematical Formulation:**

For **classification**:
$$\hat{y} = \text{mode}\{y_i : x_i \in N_k(x)\}$$

For **regression**:
$$\hat{y} = \frac{1}{k}\sum_{x_i \in N_k(x)} y_i$$

Where $N_k(x)$ is the set of k nearest neighbors of x.

### Distance Metrics

**1. Euclidean Distance (L2 Norm)**
```python
def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Example
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
dist = euclidean_distance(x1, x2)  # √27 ≈ 5.196
```

**2. Manhattan Distance (L1 Norm)**
```python
def manhattan_distance(x1, x2):
    """Calculate Manhattan distance"""
    return np.sum(np.abs(x1 - x2))

# Example
dist = manhattan_distance(x1, x2)  # |3|+|3|+|3| = 9
```

**3. Minkowski Distance (Generalized Lp Norm)**
```python
def minkowski_distance(x1, x2, p=2):
    """Calculate Minkowski distance with parameter p"""
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)

# p=1 → Manhattan, p=2 → Euclidean
```

**4. Chebyshev Distance (L∞ Norm)**
```python
def chebyshev_distance(x1, x2):
    """Maximum coordinate difference"""
    return np.max(np.abs(x1 - x2))
```

**5. Cosine Similarity**
```python
def cosine_similarity(x1, x2):
    """Measure angle between vectors"""
    dot_product = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    return dot_product / (norm1 * norm2)

def cosine_distance(x1, x2):
    return 1 - cosine_similarity(x1, x2)
```

**Distance Metric Comparison:**
| Metric | Formula | Use Case | Sensitivity |
|--------|---------|----------|-------------|
| Euclidean | √(Σ(xi-yi)²) | General purpose | Sensitive to outliers |
| Manhattan | Σ|xi-yi| | High dimensions, sparse data | Less sensitive |
| Minkowski | (Σ|xi-yi|^p)^(1/p) | Flexible (tune p) | Depends on p |
| Cosine | 1 - (x·y)/(‖x‖‖y‖) | Text, high-dimensional | Magnitude invariant |

### Choosing k

**Small k (e.g., k=1):**
- ✅ Low bias, high variance
- ✅ Captures local patterns
- ❌ Sensitive to noise
- ❌ Complex decision boundary

**Large k:**
- ✅ High bias, low variance
- ✅ Smooth decision boundary
- ❌ May miss local patterns
- ❌ Computationally expensive

**Optimal k Selection:**
```python
def find_optimal_k(X, y, k_range, cv=5):
    """Find optimal k using cross-validation"""
    cv_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    optimal_k = k_range[np.argmax(cv_scores)]
    return optimal_k, cv_scores
```

### Weighted KNN

**Problem:** All neighbors contribute equally

**Solution:** Weight by inverse distance
```python
def weighted_knn_predict(X_train, y_train, x_test, k=5):
    """Weighted KNN - closer neighbors have more influence"""
    distances = np.array([euclidean_distance(x_train, x_test) 
                          for x_train in X_train])
    
    # Get k nearest neighbors
    k_indices = np.argsort(distances)[:k]
    k_distances = distances[k_indices]
    k_labels = y_train[k_indices]
    
    # Weight by inverse distance (add small epsilon to avoid division by zero)
    weights = 1 / (k_distances + 1e-10)
    
    # Weighted vote
    unique_labels = np.unique(k_labels)
    weighted_votes = {}
    
    for label in unique_labels:
        mask = k_labels == label
        weighted_votes[label] = np.sum(weights[mask])
    
    return max(weighted_votes, key=weighted_votes.get)
```

---

## 📚 Efficient Search Structures

### 7.3.2 KD-Trees (k-dimensional Trees)

**Problem:** Brute-force KNN is O(n) per query - too slow for large datasets

**Solution:** KD-Tree organizes points in k-dimensional space for O(log n) queries

**KD-Tree Construction:**
```
Algorithm:
1. Choose axis (cycle through dimensions or max variance)
2. Find median point along that axis
3. Make median the root/subtree root
4. Recursively build left and right subtrees

Example (2D points):
Points: [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]

Step 1: Split on x-axis (median x=7)
        Root: (7,2)
        
Step 2: Left subtree (x<7), split on y-axis
        Left: (5,4)
        
Step 3: Right subtree (x>7), split on y-axis
        Right: (9,6)

Final Tree:
          (7,2) [x-split]
         /     \
    (5,4)     (9,6) [y-split]
    /   \     /
(2,3) (4,7) (8,1) [y-split]
```

**KD-Tree Implementation from Scratch:**
```python
class KDTreeNode:
    """Node in a KD-Tree"""
    def __init__(self, point, axis, left=None, right=None):
        self.point = point  # Data point
        self.axis = axis    # Splitting axis
        self.left = left    # Left subtree
        self.right = right  # Right subtree

class KDTree:
    """KD-Tree for efficient nearest neighbor search"""
    
    def __init__(self, points):
        self.k = len(points[0])  # Dimensionality
        self.root = self._build_tree(points, depth=0)
    
    def _build_tree(self, points, depth):
        """Recursively build KD-Tree"""
        if not points:
            return None
        
        # Choose axis (cycle through dimensions)
        axis = depth % self.k
        
        # Sort by axis and find median
        points_sorted = sorted(points, key=lambda x: x[axis])
        median_idx = len(points) // 2
        
        # Create node and recursively build subtrees
        return KDTreeNode(
            point=points_sorted[median_idx],
            axis=axis,
            left=self._build_tree(points_sorted[:median_idx], depth + 1),
            right=self._build_tree(points_sorted[median_idx + 1:], depth + 1)
        )
    
    def nearest_neighbor(self, query_point):
        """Find nearest neighbor to query point"""
        best = [None, float('inf')]  # [best_point, best_distance]
        self._search(self.root, query_point, best)
        return best[0]
    
    def _search(self, node, query, best):
        """Recursive search with pruning"""
        if node is None:
            return
        
        # Calculate distance to current node
        dist = euclidean_distance(np.array(node.point), np.array(query))
        if dist < best[1]:
            best[0] = node.point
            best[1] = dist
        
        # Determine which subtree to search first
        axis = node.axis
        diff = query[axis] - node.point[axis]
        
        if diff <= 0:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left
        
        # Search closer subtree
        self._search(first, query, best)
        
        # Check if we need to search farther subtree
        if abs(diff) < best[1]:
            self._search(second, query, best)
```

**Using KD-Trees in Practice:**
```python
from sklearn.neighbors import KDTree
import numpy as np

# Create sample data
X = np.random.rand(1000, 5)  # 1000 points in 5D

# Build KD-Tree
tree = KDTree(X, leaf_size=40)

# Query for nearest neighbors
query_point = np.random.rand(1, 5)
dist, ind = tree.query(query_point, k=5)

print(f"Nearest neighbor indices: {ind[0]}")
print(f"Distances: {dist[0]}")

# Radius search
neighbors_within_radius = tree.query_radius(query_point, r=0.5)
```

### Ball Trees

**When to Use Ball Trees over KD-Trees:**
- High-dimensional data (d > 20)
- Data with angular relationships
- When Euclidean distance is not ideal

**Ball Tree Concept:**
```
Ball Tree partitions data into hyperspheres (balls)
instead of hyperrectangles like KD-Trees

        Root Ball (contains all points)
              /       \
             /         \
      Ball A          Ball B
     /     \          /     \
  Ball A1  Ball A2  Ball B1  Ball B2
```

```python
from sklearn.neighbors import BallTree

# Create Ball Tree
ball_tree = BallTree(X, leaf_size=40, metric='euclidean')

# Query
dist, ind = ball_tree.query(query_point, k=5)

# Ball Trees work better with angular distances
ball_tree_cosine = BallTree(X, metric='haversine')  # For spherical data
```

---

## 📚 Learning Vector Quantization (LVQ)

### 7.3.3 LVQ Fundamentals

**Concept:** Instead of storing all training points, learn a set of **prototype vectors** (codebook vectors) that represent each class.

**Key Idea:**
```
KNN: Store ALL training points → Memory intensive
LVQ: Learn FEW prototypes → Compact representation
```

**LVQ Training Algorithm:**
```
1. Initialize codebook vectors (randomly or from training data)
2. For each training example (x, y):
   a. Find nearest codebook vector
   b. If codebook class matches y:
      - Move codebook closer to x
   c. If codebook class differs from y:
      - Move codebook away from x
3. Repeat for multiple epochs
4. Decrease learning rate over time
```

**Update Rules:**
```
Match (correct class):
    w_i(t+1) = w_i(t) + α(t) * (x - w_i(t))

Mismatch (wrong class):
    w_i(t+1) = w_i(t) - α(t) * (x - w_i(t))

Where:
- w_i: codebook vector
- x: training example
- α: learning rate (decreases over time)
```

### LVQ Implementation from Scratch

```python
class LVQ:
    """Learning Vector Quantization classifier"""
    
    def __init__(self, n_prototypes=10, learning_rate=0.1, epochs=100):
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.prototypes = None
        self.prototype_labels = None
    
    def _initialize_prototypes(self, X, y):
        """Initialize prototypes from training data"""
        n_samples = len(X)
        indices = np.random.choice(n_samples, self.n_prototypes, replace=False)
        self.prototypes = X[indices].copy()
        
        # Assign labels based on nearest training point
        self.prototype_labels = y[indices]
    
    def _find_nearest_prototype(self, x):
        """Find index of nearest prototype"""
        distances = np.array([euclidean_distance(x, p) 
                              for p in self.prototypes])
        return np.argmin(distances)
    
    def fit(self, X, y):
        """Train LVQ model"""
        self._initialize_prototypes(X, y)
        
        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X))
            
            # Decrease learning rate over time
            alpha = self.learning_rate * (1 - epoch / self.epochs)
            
            for idx in indices:
                x, label = X[idx], y[idx]
                
                # Find nearest prototype
                nearest_idx = self._find_nearest_prototype(x)
                
                # Update prototype
                if self.prototype_labels[nearest_idx] == label:
                    # Move closer
                    self.prototypes[nearest_idx] += alpha * (x - self.prototypes[nearest_idx])
                else:
                    # Move away
                    self.prototypes[nearest_idx] -= alpha * (x - self.prototypes[nearest_idx])
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        predictions = []
        for x in X:
            nearest_idx = self._find_nearest_prototype(x)
            predictions.append(self.prototype_labels[nearest_idx])
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy"""
        return np.mean(self.predict(X) == y)
```

### LVQ Variants

**LVQ1:** Basic algorithm described above

**LVQ2.1:** Updates two nearest prototypes (one from each class)

**LVQ3:** Combines LVQ1 and LVQ2.1 with additional stability term

```python
class LVQ2_1:
    """LVQ2.1 - Updates two nearest prototypes"""
    
    def __init__(self, n_prototypes=10, learning_rate=0.1, epochs=100, window=0.2):
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.window = window  # Window for simultaneous update
        self.prototypes = None
        self.prototype_labels = None
    
    def fit(self, X, y):
        """Train LVQ2.1 model"""
        self._initialize_prototypes(X, y)
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            alpha = self.learning_rate * (1 - epoch / self.epochs)
            
            for idx in indices:
                x, label = X[idx], y[idx]
                
                # Find two nearest prototypes
                distances = np.array([euclidean_distance(x, p) 
                                      for p in self.prototypes])
                nearest_indices = np.argsort(distances)[:2]
                
                i, j = nearest_indices
                d_i, d_j = distances[i], distances[j]
                
                # Check if one is correct class and one is wrong
                if (self.prototype_labels[i] == label and 
                    self.prototype_labels[j] != label):
                    correct, wrong = i, j
                elif (self.prototype_labels[j] == label and 
                      self.prototype_labels[i] != label):
                    correct, wrong = j, i
                else:
                    continue  # Both same class, skip
                
                # Check window condition
                min_d = min(d_i, d_j)
                max_d = max(d_i, d_j)
                if max_d > min_d * (1 + self.window) / (1 - self.window):
                    continue
                
                # Update both prototypes
                self.prototypes[correct] += alpha * (x - self.prototypes[correct])
                self.prototypes[wrong] -= alpha * (x - self.prototypes[wrong])
        
        return self
```

---

## 💻 Implementation from Scratch: Complete KNN

```python
import numpy as np
from collections import Counter
from typing import Tuple, List, Union

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implemented from scratch.
    
    Supports:
    - Multiple distance metrics
    - Weighted voting
    - KD-Tree for efficient search
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 metric: str = 'euclidean',
                 weights: str = 'uniform',
                 use_kdtree: bool = False):
        """
        Initialize KNN Classifier.
        
        Args:
            n_neighbors: Number of neighbors to use
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski', 'cosine')
            weights: Weight function ('uniform' or 'distance')
            use_kdtree: Whether to use KD-Tree for efficient search
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.use_kdtree = use_kdtree
        self.X_train = None
        self.y_train = None
        self.kdtree = None
    
    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate distance between two points based on selected metric"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            p = 3  # Can be parameterized
            return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)
        elif self.metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm1 = np.linalg.norm(x1)
            norm2 = np.linalg.norm(x2)
            return 1 - (dot_product / (norm1 * norm2 + 1e-10))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the KNN model.
        
        For KNN, 'fitting' just stores the training data.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # Build KD-Tree if requested
        if self.use_kdtree and len(X) > 100:
            self.kdtree = KDTree(X)
        
        return self
    
    def _find_neighbors(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for a single point"""
        if self.use_kdtree and self.kdtree is not None:
            # Use KD-Tree
            neighbors_idx = self.kdtree.nearest_neighbor_search(x, self.n_neighbors)
        else:
            # Brute force
            distances = np.array([self._calculate_distance(x, x_train) 
                                  for x_train in self.X_train])
            neighbors_idx = np.argsort(distances)[:self.n_neighbors]
        
        return self.X_train[neighbors_idx], self.y_train[neighbors_idx]
    
    def _predict_single(self, x: np.ndarray) -> Union[int, float]:
        """Predict class for a single sample"""
        neighbors_X, neighbors_y = self._find_neighbors(x)
        
        if self.weights == 'uniform':
            # Simple majority vote
            counter = Counter(neighbors_y)
            return counter.most_common(1)[0][0]
        else:
            # Distance-weighted vote
            distances = np.array([self._calculate_distance(x, neighbor) 
                                  for neighbor in neighbors_X])
            weights = 1 / (distances + 1e-10)  # Avoid division by zero
            
            # Weighted vote
            unique_labels = np.unique(neighbors_y)
            weighted_votes = {}
            
            for label in unique_labels:
                mask = neighbors_y == label
                weighted_votes[label] = np.sum(weights[mask])
            
            return max(weighted_votes, key=weighted_votes.get)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X"""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy on test data"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for each class"""
        X = np.array(X)
        unique_classes = np.unique(self.y_train)
        n_samples = len(X)
        n_classes = len(unique_classes)
        
        probas = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            _, neighbors_y = self._find_neighbors(x)
            
            if self.weights == 'uniform':
                counter = Counter(neighbors_y)
                for j, cls in enumerate(unique_classes):
                    probas[i, j] = counter.get(cls, 0) / self.n_neighbors
            else:
                # Weighted probabilities
                neighbors_X, _ = self._find_neighbors(x)
                distances = np.array([self._calculate_distance(x, neighbor) 
                                      for neighbor in neighbors_X])
                weights = 1 / (distances + 1e-10)
                
                for j, cls in enumerate(unique_classes):
                    mask = neighbors_y == cls
                    probas[i, j] = np.sum(weights[mask]) / np.sum(weights)
        
        return probas


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN
    knn = KNNClassifier(n_neighbors=5, metric='euclidean', weights='distance')
    knn.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = knn.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"KNN Accuracy: {accuracy:.4f}")
    
    # Find optimal k
    k_range = range(1, 21)
    accuracies = []
    
    for k in k_range:
        knn_k = KNNClassifier(n_neighbors=k)
        knn_k.fit(X_train_scaled, y_train)
        acc = knn_k.score(X_test_scaled, y_test)
        accuracies.append(acc)
    
    optimal_k = k_range[np.argmax(accuracies)]
    print(f"Optimal k: {optimal_k} with accuracy: {max(accuracies):.4f}")
```

---

## 📊 Summary Tables

### KNN Hyperparameters

| Hyperparameter | Description | Typical Values | Impact |
|----------------|-------------|----------------|--------|
| n_neighbors (k) | Number of neighbors | 3-10 | Controls bias-variance tradeoff |
| metric | Distance function | euclidean, manhattan, minkowski | Affects similarity measure |
| weights | Weighting scheme | uniform, distance | Gives more influence to closer neighbors |
| leaf_size | KD-Tree/Ball-Tree leaf size | 30-50 | Affects tree construction speed |

### Distance Metrics Comparison

| Metric | Formula | Best For | Limitations |
|--------|---------|----------|-------------|
| Euclidean | √Σ(xi-yi)² | Low-dimensional dense data | Suffers in high dimensions |
| Manhattan | Σ|xi-yi| | Sparse data, high dimensions | Less intuitive |
| Minkowski | (Σ|xi-yi|^p)^(1/p) | Flexible (tune p) | Need to choose p |
| Cosine | 1 - (x·y)/(‖x‖‖y‖) | Text, magnitude-invariant | Ignores magnitude |

### LVQ vs KNN

| Aspect | KNN | LVQ |
|--------|-----|-----|
| Training | Store all data (lazy) | Learn prototypes (eager) |
| Memory | O(n) | O(k) where k << n |
| Prediction | O(n) | O(k) |
| Accuracy | Generally higher | Slightly lower but faster |
| Interpretability | Hard to interpret | Prototypes are interpretable |

---

## 🎯 ML Applications

| Application | Description | Key Technique |
|-------------|-------------|---------------|
| **Recommendation Systems** | "Users who liked X also liked Y" | KNN with cosine similarity |
| **Image Recognition** | Classify images based on similar images | KNN with pixel distances |
| **Text Classification** | Categorize documents | KNN with TF-IDF + cosine |
| **Anomaly Detection** | Identify outliers | Distance to k-th neighbor |
| **Data Compression** | Vector quantization | LVQ codebooks |
| **Prototype Learning** | Learn representative examples | LVQ variants |

---

---

## ❓ Quick Check Questions

1. Why is KNN considered a "lazy learner"?
2. In high-dimensional spaces, why might Manhattan distance be preferred over Euclidean distance?
3. How does increasing the value of $k$ in KNN affect the model's bias and variance?
4. What is the primary computational advantage of using a KD-Tree for nearest neighbor search?
5. What is the main difference between KNN and Learning Vector Quantization (LVQ) in terms of memory usage?

---

## 📝 Answers to Quick Check

1. KNN is a **lazy learner** because it does not build an explicit model during the training phase. Instead, it simply stores the training instances and defers all computation until a prediction is requested.
2. In high dimensions, the difference between the distance to the nearest and farthest neighbor often becomes negligible for Euclidean distance (L2 norm). **Manhattan distance (L1 norm)** is generally more robust and provides better contrast between points in high-dimensional spaces.
3. Increasing $k$ **increases bias** (smoother decision boundary, more "averaging") and **decreases variance** (less sensitive to individual noisy points in the training data).
4. A **KD-Tree** reduces the search complexity from $O(n)$ (brute force) to an average of **$O(\log n)$** by partitioning the space into regions, allowing the algorithm to prune large areas of the search space that cannot possibly contain the nearest neighbor.
5. **KNN** must store every single training instance in memory ($O(n)$ space). **LVQ** only needs to store a small, fixed number of learned "prototype vectors" that represent the classes ($O(p)$ where $p << n$), making it much more memory-efficient for large datasets.

---

**Status:** ✅ Complete  
**Next:** [[04-Support-Vector-Machines]]
