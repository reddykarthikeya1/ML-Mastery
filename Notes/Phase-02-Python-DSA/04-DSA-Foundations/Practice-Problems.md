# DSA Foundations - Practice Problems

## Topic 1: Arrays and Strings

### Level 1: Basic

**1.1** Two Sum:
```python
def two_sum(nums, target):
    """Find two numbers that add up to target"""
    # Your code here
    pass
```

**1.2** Reverse String:
```python
def reverse_string(s):
    """Reverse string in-place"""
    # Your code here
    pass
```

### Level 2: Intermediate

**2.1** Maximum Subarray (Kadane's):
```python
def max_subarray(nums):
    """Find maximum sum subarray"""
    # Your code here
    pass
```

**2.2** Longest Substring Without Repeating:
```python
def length_of_longest_substring(s):
    """Find longest substring without repeating characters"""
    # Your code here
    pass
```

---

## Topic 2: Linked Lists

### Level 2: Intermediate

**2.1** Reverse Linked List:
```python
def reverse_list(head):
    """Reverse a linked list"""
    # Your code here
    pass
```

**2.2** Detect Cycle:
```python
def has_cycle(head):
    """Detect if linked list has cycle"""
    # Your code here
    pass
```

---

## Topic 3: Trees

### Level 2: Intermediate

**3.1** Binary Tree Traversals:
```python
def inorder(root):
    """Inorder traversal"""
    # Your code here
    pass

def level_order(root):
    """Level order traversal"""
    # Your code here
    pass
```

**3.2** Validate BST:
```python
def is_valid_bst(root):
    """Check if binary tree is valid BST"""
    # Your code here
    pass
```

---

## Topic 4: Dynamic Programming

### Level 2: Intermediate

**4.1** Climbing Stairs:
```python
def climb_stairs(n):
    """Number of ways to climb n stairs (1 or 2 at a time)"""
    # Your code here
    pass
```

**4.2** Coin Change:
```python
def coin_change(coins, amount):
    """Minimum coins needed for amount"""
    # Your code here
    pass
```

### Level 3: Advanced

**4.3** Longest Common Subsequence:
```python
def longest_common_subsequence(text1, text2):
    """Find length of LCS"""
    # Your code here
    pass
```

---

## Topic 5: Graphs

### Level 2: Intermediate

**5.1** BFS Traversal:
```python
def bfs(graph, start):
    """Breadth-first search traversal"""
    # Your code here
    pass
```

**5.2** Number of Islands:
```python
def num_islands(grid):
    """Count number of islands in grid"""
    # Your code here
    pass
```

---

## Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1 Two Sum
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return [-1, -1]
```

### 1.2 Reverse String
```python
def reverse_string(s):
    chars = list(s)
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return ''.join(chars)
```

### 2.1 Maximum Subarray
```python
def max_subarray(nums):
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum
```

### 4.1 Climbing Stairs
```python
def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

### 5.1 BFS
```python
def bfs(graph, start):
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
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
