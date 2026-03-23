# 1.4.6 Graph Theory

## 🎯 Quick Overview
- **Graphs**: Vertices and edges
- **Paths and connectivity**: Routes through graphs
- **Trees**: Acyclic connected graphs
- **Graph coloring**: Assigning colors to vertices
- **Foundation for**: Networks, GNNs, social network analysis

---

## 1. Graph Definitions

### Basic Terms

```
Graph G = (V, E)

V = set of vertices/nodes
E = set of edges/connections

Edge: (u, v) connects vertices u and v
```

### Types of Graphs

| Type | Description | Example |
|------|-------------|---------|
| **Undirected** | Edges have no direction | Social networks |
| **Directed** | Edges have direction | Web links |
| **Weighted** | Edges have weights | Road distances |
| **Simple** | No loops or multiple edges | Most applications |

---

## 2. Degree of Vertices

### Definition

```
deg(v) = number of edges incident to v

For directed graphs:
- in-degree: edges coming in
- out-degree: edges going out
```

### Handshaking Lemma

```
For undirected graphs:
Σ deg(v) = 2|E|

Corollary: Even number of vertices have odd degree
```

### Examples

```
    A -- B
    |    |
    C -- D -- E

deg(A) = 2, deg(B) = 2, deg(C) = 2, deg(D) = 3, deg(E) = 1
Sum = 10 = 2 × 5 edges ✓
```

---

## 3. Paths and Connectivity

### Path

```
Sequence of adjacent vertices

Simple path: No repeated vertices
```

### Cycle

```
Path that starts and ends at same vertex

Simple cycle: No repeated vertices (except start/end)
```

### Connected

```
Undirected: Path exists between any two vertices

Directed: Strongly connected if path exists in both directions
```

### Connected Components

```
Maximal connected subgraphs

Number of components indicates how "fragmented" graph is
```

---

## 4. Graph Representations

### Adjacency Matrix

```
n × n matrix where A[i][j] = 1 if edge from i to j

Space: O(n²)
Edge lookup: O(1)
Iterate neighbors: O(n)
```

### Adjacency List

```
Array of lists, each list contains neighbors

Space: O(V + E)
Edge lookup: O(degree)
Iterate neighbors: O(degree)
```

### Comparison

| Representation | Space | Edge Lookup | Neighbor Iteration |
|---------------|-------|-------------|-------------------|
| Adjacency Matrix | O(V²) | O(1) | O(V) |
| Adjacency List | O(V+E) | O(degree) | O(degree) |

---

## 5. Trees

### Definition

**Tree:** Connected acyclic (no cycles) undirected graph

### Properties

```
For tree with n vertices:
- |E| = n - 1
- Unique path between any two vertices
- Adding any edge creates a cycle
- Removing any edge disconnects
```

### Rooted Tree Terms

```
Root: Top node
Parent: Node above
Child: Node below
Leaf: No children
Internal: Has children
Height: Longest path to leaf
Depth: Distance from root
```

### Binary Tree

```
Each node has at most 2 children

Full binary tree: Every node has 0 or 2 children
Complete binary tree: All levels filled except possibly last
```

---

## 6. Spanning Trees

### Definition

**Spanning Tree:** Subgraph that is a tree containing all vertices

### Minimum Spanning Tree (MST)

**Spanning tree with minimum total edge weight**

### Kruskal's Algorithm

```
1. Sort edges by weight (ascending)
2. Add edges in order, skip if creates cycle
3. Stop when n-1 edges added

Uses Union-Find data structure
```

### Prim's Algorithm

```
1. Start from arbitrary vertex
2. Always add minimum weight edge connecting tree to non-tree vertex
3. Repeat until all vertices included

Uses priority queue
```

---

## 7. Graph Coloring

### Proper Coloring

```
Adjacent vertices have different colors

k-coloring: Uses at most k colors
```

### Chromatic Number

```
χ(G) = minimum number of colors needed

Examples:
- Bipartite graph: χ = 2
- Complete graph Kₙ: χ = n
- Tree: χ = 2
```

### Applications

```
- Scheduling (time slots)
- Register allocation (CPU registers)
- Sudoku solving
- Frequency assignment
```

---

## 8. Special Graphs

### Complete Graph Kₙ

```
Every pair of vertices connected

|E| = n(n-1)/2
```

### Bipartite Graph

```
Vertices can be split into two sets
All edges go between sets

Characterization: No odd cycles
```

### Planar Graph

```
Can be drawn without edge crossings

Euler's formula: V - E + F = 2
```

---

## 💻 Python Code Examples

```python
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# === Graph Representation ===

class Graph:
    """Undirected graph using adjacency list"""
    
    def __init__(self, n):
        self.n = n
        self.adj = defaultdict(list)
    
    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)
    
    def neighbors(self, v):
        return self.adj[v]
    
    def degree(self, v):
        return len(self.adj[v])
    
    def bfs(self, start):
        """Breadth-first search"""
        visited = [False] * self.n
        queue = deque([start])
        visited[start] = True
        result = [start]
        
        while queue:
            v = queue.popleft()
            for neighbor in self.adj[v]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    result.append(neighbor)
        
        return result
    
    def dfs(self, start):
        """Depth-first search"""
        visited = [False] * self.n
        result = []
        
        def dfs_helper(v):
            visited[v] = True
            result.append(v)
            for neighbor in self.adj[v]:
                if not visited[neighbor]:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return result
    
    def is_connected(self):
        """Check if graph is connected"""
        visited = [False] * self.n
        start = next(iter(self.adj.keys()))
        
        queue = deque([start])
        visited[start] = True
        count = 1
        
        while queue:
            v = queue.popleft()
            for neighbor in self.adj[v]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == self.n

# Example usage
print("Graph Operations")
print("=" * 40)

g = Graph(5)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(2, 3)
g.add_edge(3, 4)

print(f"BFS from 0: {g.bfs(0)}")
print(f"DFS from 0: {g.dfs(0)}")
print(f"Is connected: {g.is_connected()}")

# === Handshaking Lemma Verification ===

def verify_handshaking(n, edges):
    """Verify handshaking lemma"""
    
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    sum_degrees = sum(degrees)
    twice_edges = 2 * len(edges)
    
    print(f"\nHandshaking Lemma:")
    print(f"Sum of degrees: {sum_degrees}")
    print(f"2 × |E|: {twice_edges}")
    print(f"Match: {sum_degrees == twice_edges}")
    
    # Count odd degree vertices
    odd_count = sum(1 for d in degrees if d % 2 == 1)
    print(f"Vertices with odd degree: {odd_count} (should be even)")

edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
verify_handshaking(5, edges)

# === Kruskal's MST Algorithm ===

class UnionFind:
    """Union-Find data structure for Kruskal's"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True

def kruskal_mst(n, edges):
    """Kruskal's MST algorithm"""
    
    # Sort by weight
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    mst = []
    mst_weight = 0
    
    for u, v, w in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            mst_weight += w
            
            if len(mst) == n - 1:
                break
    
    return mst, mst_weight

# Example
edges = [
    (0, 1, 4), (0, 2, 3), (1, 2, 1),
    (1, 3, 2), (2, 3, 4), (3, 4, 2)
]

mst, weight = kruskal_mst(5, edges)
print(f"\nKruskal's MST:")
print(f"Edges: {mst}")
print(f"Total weight: {weight}")

# === Graph Coloring (Greedy) ===

def greedy_coloring(n, edges):
    """Greedy graph coloring"""
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    colors = {}
    
    for vertex in range(n):
        # Find colors used by neighbors
        neighbor_colors = {colors[neighbor] 
                          for neighbor in adj[vertex] 
                          if neighbor in colors}
        
        # Find smallest available color
        color = 0
        while color in neighbor_colors:
            color += 1
        
        colors[vertex] = color
    
    return colors, max(colors.values()) + 1

# Example
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
colors, n_colors = greedy_coloring(5, edges)

print(f"\nGraph Coloring:")
print(f"Colors: {colors}")
print(f"Chromatic number (upper bound): {n_colors}")

# === Visualization with NetworkX ===

def visualize_graph():
    """Visualize graph using NetworkX"""
    
    # Create graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Basic graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=15, ax=axes[0])
    axes[0].set_title('Basic Graph')
    
    # With edge weights
    G_weighted = G.copy()
    edge_weights = {(0, 1): 4, (0, 2): 3, (1, 3): 2, (2, 3): 4, (3, 4): 2}
    nx.draw(G_weighted, pos, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=15, ax=axes[1])
    nx.draw_networkx_edge_labels(G_weighted, pos, edge_labels=edge_weights, ax=axes[1])
    axes[1].set_title('Weighted Graph')
    
    plt.tight_layout()
    plt.show()

visualize_graph()

# === BFS vs DFS Paths ===

def compare_search_algorithms():
    """Compare BFS and DFS"""
    
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (0, 4)])
    
    bfs_tree = nx.bfs_tree(G, 0)
    dfs_tree = nx.dfs_tree(G, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # BFS tree
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=15, alpha=0.5, ax=axes[0])
    nx.draw(bfs_tree, pos, with_labels=True, node_color='red',
            node_size=500, font_size=15, ax=axes[0])
    axes[0].set_title('BFS Tree from 0')
    
    # DFS tree
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=15, alpha=0.5, ax=axes[1])
    nx.draw(dfs_tree, pos, with_labels=True, node_color='green',
            node_size=500, font_size=15, ax=axes[1])
    axes[1].set_title('DFS Tree from 0')
    
    plt.tight_layout()
    plt.show()

compare_search_algorithms()
```

---

## 📊 Summary Tables

### Graph Types

| Type | Description | |E| Formula |
|------|-------------|-------------|
| **Complete Kₙ** | All pairs connected | n(n-1)/2 |
| **Tree** | Connected, acyclic | n-1 |
| **Bipartite** | Two sets, cross edges only | ≤ n²/4 |
| **Cycle Cₙ** | Single cycle | n |

### Algorithm Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| BFS | O(V+E) | O(V) |
| DFS | O(V+E) | O(V) |
| Kruskal's MST | O(E log E) | O(V) |
| Prim's MST | O(E log V) | O(V) |
| Greedy Coloring | O(V + E) | O(V) |

---

## 🎯 ML Applications

| Application | Graph Theory Concept |
|-------------|---------------------|
| **GNNs** | Graph convolutions |
| **Social Networks** | Centrality, communities |
| **Knowledge Graphs** | Entity relationships |
| **Recommendation** | Bipartite matching |
| **Clustering** | Graph partitioning |

---

**Status:** ✅ Complete
**Next:** Boolean Algebra
