# 5.7 Graphs

## 🎯 Quick Overview
- **Graph**: Vertices connected by edges
- **Traversal**: BFS, DFS
- **Shortest Path**: Dijkstra, Bellman-Ford
- **Foundation for**: Networks, social graphs, knowledge graphs, GNNs

---

## 1. Graph Representation

### Adjacency List

```python
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        """Add edge (undirected)"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def add_edge_directed(self, u, v):
        """Add directed edge"""
        self.graph[u].append(v)
    
    def remove_edge(self, u, v):
        """Remove edge"""
        self.graph[u].remove(v)
        self.graph[v].remove(u)
    
    def neighbors(self, v):
        """Get neighbors"""
        return self.graph[v]
    
    def __str__(self):
        return '\n'.join(f"{k}: {v}" for k, v in self.graph.items())
```

### Adjacency Matrix

```python
class GraphMatrix:
    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]
    
    def add_edge(self, u, v):
        self.matrix[u][v] = 1
        self.matrix[v][u] = 1  # For undirected
    
    def has_edge(self, u, v):
        return self.matrix[u][v] == 1
    
    def neighbors(self, v):
        return [i for i in range(self.n) if self.matrix[v][i] == 1]
```

### Weighted Graph

```python
class WeightedGraph:
    def __init__(self):
        self.graph = defaultdict(dict)
    
    def add_edge(self, u, v, weight):
        self.graph[u][v] = weight
        self.graph[v][u] = weight  # For undirected
    
    def get_neighbors(self, u):
        return self.graph[u].items()
```

---

## 2. Graph Traversal

### BFS (Breadth-First Search)

```python
from collections import deque

def bfs(graph, start):
    """BFS traversal"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def bfs_shortest_path(graph, start, end):
    """BFS to find shortest path"""
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        vertex, path = queue.popleft()
        
        if vertex == end:
            return path
        
        if vertex not in visited:
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    
    return None

def bfs_levels(graph, root):
    """BFS level by level"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node)
            for neighbor in graph[node]:
                queue.append(neighbor)
        
        result.append(level)
    
    return result
```

### DFS (Depth-First Search)

```python
def dfs_recursive(graph, start, visited=None):
    """DFS recursive"""
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result

def dfs_iterative(graph, start):
    """DFS iterative using stack"""
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        vertex = stack.pop()
        
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            stack.extend(reversed(graph[vertex]))
    
    return result

def dfs_paths(graph, start, end, path=None):
    """Find all paths from start to end"""
    if path is None:
        path = []
    
    path = path + [start]
    
    if start == end:
        return [path]
    
    paths = []
    for neighbor in graph[start]:
        if neighbor not in path:
            new_paths = dfs_paths(graph, neighbor, end, path)
            paths.extend(new_paths)
    
    return paths
```

---

## 3. Graph Algorithms

### Cycle Detection

```python
def has_cycle_undirected(graph):
    """Detect cycle in undirected graph"""
    visited = set()
    
    def dfs(v, parent):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                return True
        return False
    
    for vertex in graph:
        if vertex not in visited:
            if dfs(vertex, None):
                return True
    return False

def has_cycle_directed(graph):
    """Detect cycle in directed graph"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}
    
    def dfs(v):
        color[v] = GRAY
        for neighbor in graph[v]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[v] = BLACK
        return False
    
    for vertex in graph:
        if color[vertex] == WHITE:
            if dfs(vertex):
                return True
    return False
```

### Topological Sort

```python
def topological_sort(graph):
    """Topological sort using DFS"""
    visited = set()
    stack = []
    
    def dfs(v):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(v)
    
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)
    
    return stack[::-1]

def topological_sort_kahn(graph):
    """Topological sort using Kahn's algorithm (BFS)"""
    from collections import deque
    
    # Calculate in-degree
    in_degree = defaultdict(int)
    for vertex in graph:
        for neighbor in graph[vertex]:
            in_degree[neighbor] += 1
    
    # Queue with 0 in-degree
    queue = deque([v for v in graph if in_degree[v] == 0])
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(graph) else []
```

### Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph, start):
    """Dijkstra's shortest path algorithm"""
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current_vertex = heapq.heappop(pq)
        
        if current_dist > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

def dijkstra_with_path(graph, start, end):
    """Dijkstra with path reconstruction"""
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    previous = {vertex: None for vertex in graph}
    pq = [(0, start)]
    
    while pq:
        current_dist, current_vertex = heapq.heappop(pq)
        
        if current_dist > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = previous[current]
    
    return distances[end], path[::-1]
```

### Union-Find (Disjoint Set Union)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True

def kruskal_mst(n, edges):
    """Kruskal's MST algorithm"""
    edges.sort(key=lambda x: x[2])  # Sort by weight
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
```

---

## 4. Graph Problems

### Number of Islands

```python
def num_islands(grid):
    """Count number of islands"""
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] == '0' or (r, c) in visited):
            return
        
        visited.add((r, c))
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1' and (i, j) not in visited:
                dfs(i, j)
                count += 1
    
    return count
```

### Clone Graph

```python
def clone_graph(node):
    """Clone undirected graph"""
    if not node:
        return None
    
    clones = {}
    
    def dfs(n):
        if n in clones:
            return clones[n]
        
        clone = Node(n.val)
        clones[n] = clone
        
        for neighbor in n.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```

### Course Schedule

```python
def can_finish(num_courses, prerequisites):
    """Check if all courses can be finished"""
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    visited = set()
    rec_stack = set()
    
    def has_cycle(v):
        visited.add(v)
        rec_stack.add(v)
        
        for neighbor in graph[v]:
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(v)
        return False
    
    for i in range(num_courses):
        if i not in visited:
            if has_cycle(i):
                return False
    
    return True
```

---

## 💻 Python Code Examples

```python
# === Word Ladder ===

def ladder_length(begin_word, end_word, word_list):
    """Shortest transformation sequence"""
    from collections import deque
    
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    
    queue = deque([(begin_word, 1)])
    visited = {begin_word}
    
    while queue:
        word, length = queue.popleft()
        
        if word == end_word:
            return length
        
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, length + 1))
    
    return 0

# === Network Delay Time ===

def network_delay_time(times, n, k):
    """Time for signal to reach all nodes"""
    graph = defaultdict(dict)
    for u, v, w in times:
        graph[u][v] = w
    
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0
    pq = [(0, k)]
    
    while pq:
        dist, node = heapq.heappop(pq)
        
        if dist > distances[node]:
            continue
        
        for neighbor, weight in graph[node].items():
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    max_dist = max(distances.values())
    return max_dist if max_dist < float('inf') else -1

# === Pacific Atlantic Water Flow ===

def pacific_atlantic(heights):
    """Find cells that can flow to both oceans"""
    if not heights:
        return []
    
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    
    def dfs(r, c, visited):
        visited.add((r, c))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                (nr, nc) not in visited and
                heights[nr][nc] >= heights[r][c]):
                dfs(nr, nc, visited)
    
    # DFS from Pacific (top and left)
    for i in range(rows):
        dfs(i, 0, pacific)
    for j in range(cols):
        dfs(0, j, pacific)
    
    # DFS from Atlantic (bottom and right)
    for i in range(rows):
        dfs(i, cols - 1, atlantic)
    for j in range(cols):
        dfs(rows - 1, j, atlantic)
    
    # Intersection
    result = []
    for i in range(rows):
        for j in range(cols):
            if (i, j) in pacific and (i, j) in atlantic:
                result.append([i, j])
    
    return result
```

---

## 📊 Summary Tables

### Graph Representations

| Representation | Space | Edge Lookup | Iterate Neighbors |
|---------------|-------|-------------|-------------------|
| Adjacency List | O(V+E) | O(degree) | O(degree) |
| Adjacency Matrix | O(V²) | O(1) | O(V) |

### Graph Algorithms

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| BFS | O(V+E) | O(V) | Shortest path (unweighted) |
| DFS | O(V+E) | O(V) | Cycle detection, topological sort |
| Dijkstra | O((V+E) log V) | O(V) | Shortest path (weighted) |
| Union-Find | O(E α(V)) | O(V) | Connected components, MST |
| Kruskal | O(E log E) | O(V) | Minimum spanning tree |

---

## 🎯 ML Applications

| Graph Concept | ML Application |
|---------------|----------------|
| Graph Traversal | Knowledge graph navigation |
| Shortest Path | Recommendation paths |
| Connected Components | Community detection |
| Topological Sort | Dependency resolution |
| Graph Neural Networks | Node classification |

---

**Status:** ✅ Complete
**Next:** Sorting and Searching
