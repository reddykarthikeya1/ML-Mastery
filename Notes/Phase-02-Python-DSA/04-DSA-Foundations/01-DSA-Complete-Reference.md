# 5.1-5.10 DSA Foundations - Complete Reference

## 🎯 Quick Overview
- **Complexity**: Analyze algorithm efficiency
- **Data Structures**: Organize and store data
- **Algorithms**: Solve computational problems
- **Foundation for**: Efficient coding, technical interviews, ML engineering

---

## Part 1: Complexity Analysis (5.1)

### Big O Notation

```
Big O: Upper bound (worst case)
Omega (Ω): Lower bound (best case)
Theta (Θ): Tight bound (average case)
```

### Common Complexities

```
O(1)         - Constant time
O(log n)     - Logarithmic
O(n)         - Linear
O(n log n)   - Linearithmic
O(n²)        - Quadratic
O(2ⁿ)        - Exponential
O(n!)        - Factorial
```

### Analyzing Code

```python
# O(1) - Constant
def get_first(arr):
    return arr[0]

# O(n) - Linear
def find_max(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

# O(n²) - Quadratic
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates

# O(log n) - Logarithmic
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

---

## Part 2: Arrays and Strings (5.2)

### Two-Pointer Technique

```python
def two_sum_sorted(arr, target):
    """Find two numbers that sum to target"""
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]

def reverse_string(s):
    """Reverse string in-place"""
    chars = list(s)
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return ''.join(chars)
```

### Sliding Window

```python
def max_sum_subarray(arr, k):
    """Maximum sum of k consecutive elements"""
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

def longest_unique_substring(s):
    """Longest substring without repeating characters"""
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

---

## Part 3: Linked Lists (5.3)

### Singly Linked List

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        """Add to end"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def reverse(self):
        """Reverse linked list"""
        prev = None
        current = self.head
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        self.head = prev
    
    def detect_cycle(self):
        """Floyd's cycle detection"""
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

---

## Part 4: Stacks and Queues (5.4)

### Stack Implementation

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
    
    def peek(self):
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0

# Valid parentheses
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    
    return not stack
```

### Queue Implementation

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        return self.items.popleft()
    
    def is_empty(self):
        return len(self.items) == 0

# BFS using queue
def bfs(graph, start):
    visited = set()
    queue = Queue()
    queue.enqueue(start)
    visited.add(start)
    result = []
    
    while not queue.is_empty():
        vertex = queue.dequeue()
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.enqueue(neighbor)
    
    return result
```

---

## Part 5: Hash Tables (5.5)

### Hash Table Implementation

```python
class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))
    
    def get(self, key):
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None
    
    def remove(self, key):
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                return

# Two Sum using hash table
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return [-1, -1]
```

---

## Part 6: Trees and BSTs (5.6)

### Binary Tree Traversals

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder(root):
    """Left, Root, Right"""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):
    """Root, Left, Right"""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root):
    """Left, Right, Root"""
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

def level_order(root):
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
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### Binary Search Tree

```python
class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        def insert_node(node, val):
            if not node:
                return TreeNode(val)
            if val < node.val:
                node.left = insert_node(node.left, val)
            else:
                node.right = insert_node(node.right, val)
            return node
        
        self.root = insert_node(self.root, val)
    
    def search(self, val):
        def search_node(node, val):
            if not node:
                return False
            if node.val == val:
                return True
            elif val < node.val:
                return search_node(node.left, val)
            else:
                return search_node(node.right, val)
        
        return search_node(self.root, val)
```

### Heap (Priority Queue)

```python
import heapq

# Min Heap
min_heap = []
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 1)
heapq.heappush(min_heap, 2)
smallest = heapq.heappop(min_heap)  # 1

# Max Heap (negate values)
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
heapq.heappush(max_heap, -2)
largest = -heapq.heappop(max_heap)  # 3

# Top K elements
def top_k_frequent(nums, k):
    from collections import Counter
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

---

## Part 7: Graphs (5.7)

### Graph Representation

```python
from collections import defaultdict, deque

# Adjacency List
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # For undirected
    
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def dfs(self, start):
        visited = set()
        result = []
        
        def dfs_helper(vertex):
            visited.add(vertex)
            result.append(vertex)
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return result
```

### Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph, start):
    """Shortest path from start to all nodes"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_dist > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

---

## Part 8: Sorting and Searching (5.8)

### Sorting Algorithms

```python
def quick_sort(arr):
    """O(n log n) average, O(n²) worst"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def merge_sort(arr):
    """O(n log n) all cases"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Binary Search

```python
def binary_search(arr, target):
    """Find target in sorted array"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_leftmost(arr, target):
    """Find leftmost occurrence"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left if left < len(arr) and arr[left] == target else -1
```

---

## Part 9: Dynamic Programming (5.9)

### DP Patterns

```python
# Fibonacci (Memoization)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Fibonacci (Tabulation)
def fibonacci_tab(n):
    if n <= 1:
        return n
    dp = [0, 1]
    for i in range(2, n+1):
        dp.append(dp[i-1] + dp[i-2])
    return dp[n]

# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# Longest Common Subsequence
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

---

## Part 10: Algorithm Design Patterns (5.10)

### Greedy Algorithms

```python
def activity_selection(activities):
    """Select maximum non-overlapping activities"""
    activities.sort(key=lambda x: x[1])  # Sort by end time
    result = [activities[0]]
    
    for activity in activities[1:]:
        if activity[0] >= result[-1][1]:
            result.append(activity)
    
    return result

def coin_change_greedy(coins, amount):
    """Greedy coin change (works for canonical systems)"""
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        count += amount // coin
        amount %= coin
    return count if amount == 0 else -1
```

### Divide and Conquer

```python
def merge_sort(arr):
    """Classic divide and conquer"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def closest_pair(points):
    """Find closest pair of points (divide and conquer)"""
    # Simplified version
    def distance(p1, p2):
        return ((p1[0] - p2[0])**2 + **(p1[1] - p2[1])2)**0.5
    
    min_dist = float('inf')
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            min_dist = min(min_dist, distance(points[i], points[j]))
    return min_dist
```

### Backtracking

```python
def n_queens(n):
    """N-Queens problem"""
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or \
               board[i] - i == col - row or \
               board[i] + i == col + row:
                return False
        return True
    
    def solve(row):
        if row == n:
            result.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(row + 1)
                board[row] = -1
    
    board = [-1] * n
    result = []
    solve(0)
    return result

def permutations(nums):
    """Generate all permutations"""
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            backtrack(path + [remaining[i]], remaining[:i] + remaining[i+1:])
    
    result = []
    backtrack([], nums)
    return result
```

---

## 📊 Summary Tables

### Complexity Comparison

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| BFS | O(V+E) | O(V+E) | O(V+E) | O(V) |
| DFS | O(V+E) | O(V+E) | O(V+E) | O(V) |

### Data Structure Operations

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) |
| Hash Table | N/A | O(1) | O(1) | O(1) |
| BST | O(log n) | O(log n) | O(log n) | O(log n) |
| Heap | O(1) | O(n) | O(log n) | O(log n) |

---

## 🎯 ML Applications

| DSA Concept | ML Application |
|-------------|----------------|
| Hash Tables | Feature hashing, vocabularies |
| Trees | Decision trees, XGBoost |
| Graphs | Knowledge graphs, GNNs |
| Heaps | Priority queues, beam search |
| DP | Sequence alignment, Viterbi |
| Sorting | Data preprocessing |

---

**Status:** ✅ Complete
**Next:** ML Fundamentals
