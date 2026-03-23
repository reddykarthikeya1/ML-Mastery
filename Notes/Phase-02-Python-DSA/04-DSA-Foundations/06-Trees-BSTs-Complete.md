# 5.6 Trees and BSTs

## 🎯 Quick Overview
- **Tree**: Hierarchical data structure
- **Binary Tree**: Each node has at most 2 children
- **BST**: Ordered binary tree
- **Heap**: Complete binary tree with heap property
- **Foundation for**: Search algorithms, databases, ML models

---

## 1. Binary Tree

### Node Definition

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Tree Traversals

```python
def inorder(root):
    """Inorder: Left, Root, Right"""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):
    """Preorder: Root, Left, Right"""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root):
    """Postorder: Left, Right, Root"""
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

def level_order(root):
    """Level order (BFS)"""
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

# Iterative traversals
def inorder_iterative(root):
    """Inorder iterative using stack"""
    result = []
    stack = []
    current = root
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
    return result

def preorder_iterative(root):
    """Preorder iterative using stack"""
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
```

### Tree Properties

```python
def max_depth(root):
    """Maximum depth of binary tree"""
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def min_depth(root):
    """Minimum depth of binary tree"""
    if not root:
        return 0
    if not root.left:
        return 1 + min_depth(root.right)
    if not root.right:
        return 1 + min_depth(root.left)
    return 1 + min(min_depth(root.left), min_depth(root.right))

def is_balanced(root):
    """Check if tree is balanced"""
    def check_height(node):
        if not node:
            return 0
        
        left_height = check_height(node.left)
        if left_height == -1:
            return -1
        
        right_height = check_height(node.right)
        if right_height == -1:
            return -1
        
        if abs(left_height - right_height) > 1:
            return -1
        
        return 1 + max(left_height, right_height)
    
    return check_height(root) != -1

def is_symmetric(root):
    """Check if tree is symmetric"""
    def is_mirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return (t1.val == t2.val and 
                is_mirror(t1.left, t2.right) and 
                is_mirror(t1.right, t2.left))
    
    return is_mirror(root, root)
```

---

## 2. Binary Search Tree (BST)

### BST Operations

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
    
    def delete(self, val):
        def delete_node(node, val):
            if not node:
                return None
            
            if val < node.val:
                node.left = delete_node(node.left, val)
            elif val > node.val:
                node.right = delete_node(node.right, val)
            else:
                # Node to delete found
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left
                
                # Node with two children
                min_larger = get_min(node.right)
                node.val = min_larger.val
                node.right = delete_node(node.right, min_larger.val)
            
            return node
        
        def get_min(node):
            current = node
            while current.left:
                current = current.left
            return current
        
        self.root = delete_node(self.root, val)
    
    def validate(self):
        """Validate if tree is valid BST"""
        def is_valid(node, min_val, max_val):
            if not node:
                return True
            if node.val <= min_val or node.val >= max_val:
                return False
            return (is_valid(node.left, min_val, node.val) and 
                    is_valid(node.right, node.val, max_val))
        
        return is_valid(self.root, float('-inf'), float('inf'))
```

### BST Problems

```python
def lowest_common_ancestor(root, p, q):
    """Lowest common ancestor in BST"""
    current = root
    
    while current:
        if p.val < current.val and q.val < current.val:
            current = current.left
        elif p.val > current.val and q.val > current.val:
            current = current.right
        else:
            return current
    
    return None

def kth_smallest(root, k):
    """Kth smallest element in BST"""
    stack = []
    current = root
    count = 0
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        count += 1
        
        if count == k:
            return current.val
        
        current = current.right
    
    return None

def convert_sorted_array_to_bst(nums):
    """Convert sorted array to height-balanced BST"""
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = convert_sorted_array_to_bst(nums[:mid])
    root.right = convert_sorted_array_to_bst(nums[mid+1:])
    
    return root
```

---

## 3. Heaps and Priority Queues

### Heap Implementation

```python
import heapq

# Min heap
min_heap = []
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 1)
heapq.heappush(min_heap, 2)
smallest = heapq.heappop(min_heap)  # 1

# Max heap (negate values)
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
heapq.heappush(max_heap, -2)
largest = -heapq.heappop(max_heap)  # 3

# Heap with custom comparison
heap = []
heapq.heappush(heap, (priority, item))
```

### Heap Problems

```python
def top_k_frequent(nums, k):
    """Top K frequent elements"""
    from collections import Counter
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

def merge_k_sorted_lists(lists):
    """Merge K sorted linked lists"""
    min_heap = []
    
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    current = dummy
    
    while min_heap:
        val, i, node = heapq.heappop(min_heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(min_heap, (node.next.val, i, node.next))
    
    return dummy.next

def find_median_from_data_stream():
    """Find median from data stream"""
    min_heap = []  # Larger half
    max_heap = []  # Smaller half (negated)
    
    def add_num(num):
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)
        
        # Balance
        if len(max_heap) > len(min_heap) + 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
    
    def find_median():
        if len(max_heap) == len(min_heap):
            return (-max_heap[0] + min_heap[0]) / 2
        return -max_heap[0]
    
    return add_num, find_median
```

---

## 4. Trie (Prefix Tree)

### Trie Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### Trie Problems

```python
def implement_trie():
    """Implement Trie with autocomplete"""
    class Trie:
        def __init__(self):
            self.root = TrieNode()
        
        def insert(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        def autocomplete(self, prefix):
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return []
                node = node.children[char]
            
            results = []
            self._dfs(node, prefix, results)
            return results
        
        def _dfs(self, node, path, results):
            if node.is_end:
                results.append(path)
            for char, child in node.children.items():
                self._dfs(child, path + char, results)
    
    return Trie()
```

---

## 5. Segment Tree and Fenwick Tree

### Segment Tree

```python
class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self._build(nums, 0, 0, self.n - 1)
    
    def _build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            self._build(nums, 2 * node + 1, start, mid)
            self._build(nums, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def update(self, idx, val):
        self._update(0, 0, self.n - 1, idx, val)
    
    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                self._update(2 * node + 1, start, mid, idx, val)
            else:
                self._update(2 * node + 2, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def query(self, left, right):
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return (self._query(2 * node + 1, start, mid, left, right) + 
                self._query(2 * node + 2, mid + 1, end, left, right))
```

### Fenwick Tree (Binary Indexed Tree)

```python
class FenwickTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (self.n + 1)
        for i in range(self.n):
            self.update(i, nums[i])
    
    def update(self, idx, delta):
        idx += 1  # 1-indexed
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & (-idx)
    
    def query(self, idx):
        idx += 1  # 1-indexed
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & (-idx)
        return result
    
    def query_range(self, left, right):
        return self.query(right) - self.query(left - 1)
```

---

## 💻 Python Code Examples

```python
# === Serialize and Deserialize Binary Tree ===

def serialize(root):
    """Serialize binary tree to string"""
    def preorder(node):
        if not node:
            return ['#']
        return [str(node.val)] + preorder(node.left) + preorder(node.right)
    
    return ','.join(preorder(root))

def deserialize(data):
    """Deserialize string to binary tree"""
    def preorder():
        val = next(vals)
        if val == '#':
            return None
        node = TreeNode(int(val))
        node.left = preorder()
        node.right = preorder()
        return node
    
    vals = iter(data.split(','))
    return preorder()

# === Word Search II ===

def find_words(board, words):
    """Find all words in board using Trie"""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None
    
    # Build Trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    
    # Search
    result = []
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, node):
        char = board[r][c]
        if char not in node.children:
            return
        
        node = node.children[char]
        if node.word:
            result.append(node.word)
            node.word = None  # Avoid duplicates
        
        board[r][c] = '#'  # Mark as visited
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, node)
        
        board[r][c] = char  # Restore
    
    for i in range(rows):
        for j in range(cols):
            dfs(i, j, root)
    
    return result
```

---

## 📊 Summary Tables

### Tree Traversals

| Traversal | Order | Use Case |
|-----------|-------|----------|
| Inorder | Left, Root, Right | BST gives sorted order |
| Preorder | Root, Left, Right | Copy tree, serialize |
| Postorder | Left, Right, Root | Delete tree |
| Level Order | Level by level | BFS, level-wise processing |

### BST Operations

| Operation | Time | Space |
|-----------|------|-------|
| Search | O(log n) | O(log n) |
| Insert | O(log n) | O(log n) |
| Delete | O(log n) | O(log n) |
| Min/Max | O(log n) | O(1) |

### Heap Operations

| Operation | Time | Example |
|-----------|------|---------|
| Push | O(log n) | heapq.heappush |
| Pop | O(log n) | heapq.heappop |
| Peek | O(1) | heap[0] |
| Heapify | O(n) | heapq.heapify |

---

## 🎯 ML Applications

| Tree Concept | ML Application |
|-------------|----------------|
| BST | Decision trees, indexing |
| Heap | Priority queues, beam search |
| Trie | Autocomplete, spell checker |
| Segment Tree | Range queries, feature aggregation |

---

**Status:** ✅ Complete
**Next:** Graphs
