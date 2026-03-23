# 5.4 Stacks and Queues

## 🎯 Quick Overview
- **Stack**: LIFO (Last In First Out)
- **Queue**: FIFO (First In First Out)
- **Applications**: Expression evaluation, BFS/DFS, backtracking
- **Foundation for**: Tree/graph traversals, parsing

---

## 1. Stack

### Implementation

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
    
    def size(self):
        return len(self.items)

# Using deque for better performance
from collections import deque

class Stack:
    def __init__(self):
        self.items = deque()
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
```

### Stack Problems

```python
def is_valid_parentheses(s):
    """Valid parentheses"""
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

def min_stack():
    """Stack with getMin operation"""
    stack = []
    min_stack = []
    
    def push(x):
        stack.append(x)
        if not min_stack or x <= min_stack[-1]:
            min_stack.append(x)
    
    def pop():
        if stack.pop() == min_stack[-1]:
            min_stack.pop()
    
    def get_min():
        return min_stack[-1] if min_stack else None
    
    return push, pop, get_min

def evaluate_rpn(tokens):
    """Evaluate Reverse Polish Notation"""
    stack = []
    
    for token in tokens:
        if token in '+-*/':
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]

def daily_temperatures(temperatures):
    """Daily temperatures - next warmer day"""
    result = [0] * len(temperatures)
    stack = []  # (temperature, index)
    
    for i, temp in enumerate(temperatures):
        while stack and temp > stack[-1][0]:
            prev_temp, prev_i = stack.pop()
            result[prev_i] = i - prev_i
        stack.append((temp, i))
    
    return result

def largest_rectangle_area(heights):
    """Largest rectangle in histogram"""
    stack = []
    max_area = 0
    heights.append(0)  # Sentinel
    
    for i, h in enumerate(heights):
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    heights.pop()
    return max_area
```

---

## 2. Queue

### Implementation

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        return self.items.popleft()
    
    def peek(self):
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Circular Queue
class CircularQueue:
    def __init__(self, k):
        self.k = k
        self.queue = [None] * k
        self.head = self.tail = -1
    
    def enqueue(self, value):
        if self.is_full():
            return False
        if self.is_empty():
            self.head = 0
        self.tail = (self.tail + 1) % self.k
        self.queue[self.tail] = value
        return True
    
    def dequeue(self):
        if self.is_empty():
            return False
        if self.head == self.tail:
            self.head = self.tail = -1
        else:
            self.head = (self.head + 1) % self.k
        return True
    
    def is_empty(self):
        return self.head == -1
    
    def is_full(self):
        return (self.tail + 1) % self.k == self.head
```

### Queue Problems

```python
def bfs(graph, start):
    """BFS using queue"""
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

def level_order_traversal(root):
    """Binary tree level order traversal"""
    if not root:
        return []
    
    result = []
    queue = Queue()
    queue.enqueue(root)
    
    while not queue.is_empty():
        level_size = queue.size()
        level = []
        
        for _ in range(level_size):
            node = queue.dequeue()
            level.append(node.val)
            if node.left:
                queue.enqueue(node.left)
            if node.right:
                queue.enqueue(node.right)
        
        result.append(level)
    
    return result

def sliding_window_maximum(nums, k):
    """Sliding window maximum using deque"""
    from collections import deque
    
    result = []
    dq = deque()  # Store indices
    
    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum for this window
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

---

## 3. Priority Queue (Heap)

### Implementation

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

# Heap with tuples
heap = []
heapq.heappush(heap, (1, 'first'))
heapq.heappush(heap, (3, 'third'))
heapq.heappush(heap, (2, 'second'))
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

def find_median_stream():
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

## 4. Monotonic Stack/Queue

```python
def next_greater_element(nums):
    """Next greater element for each element"""
    result = [-1] * len(nums)
    stack = []
    
    for i in range(len(nums) - 1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(nums[i])
    
    return result

def max_sliding_window(nums, k):
    """Maximum in sliding window using monotonic deque"""
    from collections import deque
    
    result = []
    dq = deque()
    
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

---

## 💻 Python Code Examples

```python
# === Implement Queue using Stacks ===

class QueueWithStacks:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []
    
    def push(self, x):
        self.in_stack.append(x)
    
    def pop(self):
        self._move()
        return self.out_stack.pop()
    
    def peek(self):
        self._move()
        return self.out_stack[-1]
    
    def _move(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

# === Basic Calculator ===

def calculate(s):
    """Basic calculator with +, -, (, )"""
    stack = []
    num = 0
    sign = 1
    result = 0
    
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char in '+-':
            result += sign * num
            num = 0
            sign = 1 if char == '+' else -1
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            result *= stack.pop()
            result += stack.pop()
    
    result += sign * num
    return result

# === Decode String ===

def decode_string(s):
    """Decode encoded string: 3[a]2[bc] -> aaabcbc"""
    stack = []
    current_num = 0
    current_str = ''
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_str, current_num))
            current_str = ''
            current_num = 0
        elif char == ']':
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char
    
    return current_str
```

---

## 📊 Summary Tables

### Stack vs Queue

| Feature | Stack | Queue |
|---------|-------|-------|
| Order | LIFO | FIFO |
| Insert | push | enqueue |
| Remove | pop | dequeue |
| Use Case | DFS, undo | BFS, scheduling |

### Heap Operations

| Operation | Time | Example |
|-----------|------|---------|
| Push | O(log n) | heapq.heappush |
| Pop | O(log n) | heapq.heappop |
| Peek | O(1) | heap[0] |
| Heapify | O(n) | heapq.heapify |

---

## 🎯 ML Applications

| Stack/Queue Concept | ML Application |
|---------------------|----------------|
| Stack | DFS, backtracking |
| Queue | BFS, batch processing |
| Priority Queue | Beam search, top-k |
| Monotonic Stack | Feature selection |

---

**Status:** ✅ Complete
**Next:** Hash Tables
