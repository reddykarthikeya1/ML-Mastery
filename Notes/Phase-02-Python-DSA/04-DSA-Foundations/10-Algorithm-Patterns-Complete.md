# 5.10 Algorithm Design Patterns

## 🎯 Quick Overview
- **Greedy**: Make locally optimal choices
- **Divide & Conquer**: Break into subproblems
- **Backtracking**: Explore all possibilities
- **Foundation for**: Algorithm design, problem solving, optimization

---

## 1. Greedy Algorithms

### Pattern

```
Make locally optimal choice at each step
Hope it leads to global optimum

When it works:
- Greedy choice property
- Optimal substructure

When it doesn't:
- Need to reconsider previous choices
```

### Activity Selection

```python
def activity_selection(activities):
    """Select maximum non-overlapping activities"""
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    
    result = [activities[0]]
    for activity in activities[1:]:
        if activity[0] >= result[-1][1]:
            result.append(activity)
    
    return result

# Example
activities = [(1, 3), (2, 5), (4, 7), (1, 8), (5, 9), (8, 10)]
print(activity_selection(activities))  # [(1, 3), (4, 7), (8, 10)]
```

### Interval Scheduling

```python
def min_meeting_rooms(intervals):
    """Minimum meeting rooms required"""
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])
    
    rooms = 0
    end_ptr = 0
    
    for start in starts:
        if start >= ends[end_ptr]:
            end_ptr += 1
        else:
            rooms += 1
    
    return rooms

def merge_intervals(intervals):
    """Merge overlapping intervals"""
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    
    return merged
```

### Fractional Knapsack

```python
def fractional_knapsack(items, capacity):
    """Fractional knapsack - can take fractions"""
    # Sort by value/weight ratio
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0
    remaining = capacity
    
    for value, weight in items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            fraction = remaining / weight
            total_value += value * fraction
            break
    
    return total_value
```

### Huffman Coding

```python
import heapq
from collections import defaultdict

def huffman_coding(text):
    """Huffman coding for text compression"""
    # Count frequencies
    freq = defaultdict(int)
    for char in text:
        freq[char] += 1
    
    # Build min heap
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Sort by code length
    codes = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[1]), p))
    return dict(codes)

# Example
text = "hello world"
codes = huffman_coding(text)
print(codes)
```

### Task Scheduling

```python
def min_time_to_complete_tasks(tasks, n):
    """Minimum time with cooldown n between same tasks"""
    from collections import Counter
    
    counts = Counter(tasks)
    max_freq = max(counts.values())
    max_count = sum(1 for v in counts.values() if v == max_freq)
    
    return max(len(tasks), (max_freq - 1) * (n + 1) + max_count)
```

---

## 2. Divide and Conquer

### Pattern

```
1. Divide problem into subproblems
2. Conquer subproblems recursively
3. Combine solutions

Time complexity: T(n) = aT(n/b) + f(n)
Master theorem applies
```

### Merge Sort

```python
def merge_sort(arr):
    """Classic divide and conquer"""
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

### Quick Sort

```python
def quick_sort(arr):
    """Divide and conquer with pivot"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
```

### Closest Pair of Points

```python
def closest_pair(points):
    """Find closest pair of points - O(n log n)"""
    def distance(p1, p2):
        return ((p1[0] - p2[0])**2 + **(p1[1] - p2[1])2)**0.5
    
    def brute_force(pts):
        min_dist = float('inf')
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                min_dist = min(min_dist, distance(pts[i], pts[j]))
        return min_dist
    
    def strip_closest(strip, d):
        min_val = d
        strip.sort(key=lambda x: x[1])
        
        for i in range(len(strip)):
            for j in range(i+1, min(i+7, len(strip))):
                min_val = min(min_val, distance(strip[i], strip[j]))
        
        return min_val
    
    def divide_and_conquer(pts_sorted_x):
        n = len(pts_sorted_x)
        if n <= 3:
            return brute_force(pts_sorted_x)
        
        mid = n // 2
        mid_point = pts_sorted_x[mid]
        
        left_min = divide_and_conquer(pts_sorted_x[:mid])
        right_min = divide_and_conquer(pts_sorted_x[mid:])
        d = min(left_min, right_min)
        
        strip = [p for p in pts_sorted_x if abs(p[0] - mid_point[0]) < d]
        return min(d, strip_closest(strip, d))
    
    points_sorted_x = sorted(points, key=lambda x: x[0])
    return divide_and_conquer(points_sorted_x)
```

### Karatsuba Multiplication

```python
def karatsuba(x, y):
    """Fast multiplication - O(n^1.585)"""
    if x < 10 or y < 10:
        return x * y
    
    n = max(len(str(x)), len(str(y)))
    half = n // 2
    
    # Split numbers
    high1, low1 = divmod(x, 10**half)
    high2, low2 = divmod(y, 10**half)
    
    # Recursive calls
    z0 = karatsuba(low1, low2)
    z2 = karatsuba(high1, high2)
    z1 = karatsuba(low1 + high1, low2 + high2) - z2 - z0
    
    return z2 * 10**(2*half) + z1 * 10**half + z0
```

---

## 3. Backtracking

### Pattern

```
1. Choose a candidate
2. Explore recursively
3. Backtrack if invalid
4. Try next candidate
```

### N-Queens

```python
def n_queens(n):
    """N-Queens problem"""
    def is_safe(board, row, col):
        for i in range(row):
            if (board[i] == col or 
                board[i] - i == col - row or 
                board[i] + i == col + row):
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

# Example
solutions = n_queens(4)
print(f"Found {len(solutions)} solutions")
```

### Permutations

```python
def permutations(nums):
    """Generate all permutations"""
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        
        for i in range(len(remaining)):
            backtrack(path + [remaining[i]], 
                     remaining[:i] + remaining[i+1:])
    
    result = []
    backtrack([], nums)
    return result

def permutations_with_duplicates(nums):
    """Permutations with duplicates"""
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    
    nums.sort()
    result = []
    backtrack([], [False] * len(nums))
    return result
```

### Subsets

```python
def subsets(nums):
    """Generate all subsets (power set)"""
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

def subsets_with_duplicates(nums):
    """Subsets with duplicates"""
    result = []
    nums.sort()
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

### Combination Sum

```python
def combination_sum(candidates, target):
    """Find all combinations that sum to target"""
    result = []
    
    def backtrack(start, path, total):
        if total == target:
            result.append(path[:])
            return
        if total > target:
            return
        
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, total + candidates[i])  # Can reuse
            path.pop()
    
    backtrack(0, [], 0)
    return result

def combination_sum_2(candidates, target):
    """Each element used once"""
    result = []
    candidates.sort()
    
    def backtrack(start, path, total):
        if total == target:
            result.append(path[:])
            return
        if total > target:
            return
        
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i-1]:
                continue
            path.append(candidates[i])
            backtrack(i + 1, path, total + candidates[i])
            path.pop()
    
    backtrack(0, [], 0)
    return result
```

### Sudoku Solver

```python
def solve_sudoku(board):
    """Solve Sudoku puzzle"""
    def is_valid(row, col, num):
        for i in range(9):
            if (board[row][i] == num or 
                board[i][col] == num or 
                board[3*(row//3) + i//3][3*(col//3) + i%3] == num):
                return False
        return True
    
    def solve():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num
                            if solve():
                                return True
                            board[row][col] = '.'
                    return False
        return True
    
    solve()
```

### Word Search

```python
def exist(board, word):
    """Word search in 2D board"""
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, idx):
        if idx == len(word):
            return True
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[idx]):
            return False
        
        temp = board[r][c]
        board[r][c] = '#'
        
        found = (dfs(r+1, c, idx+1) or
                dfs(r-1, c, idx+1) or
                dfs(r, c+1, idx+1) or
                dfs(r, c-1, idx+1))
        
        board[r][c] = temp
        return found
    
    for i in range(rows):
        for j in range(cols):
            if dfs(i, j, 0):
                return True
    
    return False
```

---

## 4. Algorithm Comparison

```python
# Problem: Find maximum subarray sum

# Brute force - O(n²)
def max_subarray_brute(nums):
    max_sum = float('-inf')
    for i in range(len(nums)):
        current_sum = 0
        for j in range(i, len(nums)):
            current_sum += nums[j]
            max_sum = max(max_sum, current_sum)
    return max_sum

# Divide and conquer - O(n log n)
def max_subarray_divide(nums):
    def helper(left, right):
        if left == right:
            return nums[left]
        
        mid = (left + right) // 2
        
        left_max = helper(left, mid)
        right_max = helper(mid + 1, right)
        
        # Cross sum
        left_sum = float('-inf')
        current = 0
        for i in range(mid, left - 1, -1):
            current += nums[i]
            left_sum = max(left_sum, current)
        
        right_sum = float('-inf')
        current = 0
        for i in range(mid + 1, right + 1):
            current += nums[i]
            right_sum = max(right_sum, current)
        
        return max(left_max, right_max, left_sum + right_sum)
    
    return helper(0, len(nums) - 1)

# Kadane's algorithm (Greedy/DP) - O(n)
def max_subarray_kadane(nums):
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum
```

---

## 💻 Python Code Examples

```python
# === Generate Parentheses ===

def generate_parentheses(n):
    """Generate all valid parentheses combinations"""
    result = []
    
    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append(path)
            return
        
        if open_count < n:
            backtrack(path + '(', open_count + 1, close_count)
        
        if close_count < open_count:
            backtrack(path + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result

# === Letter Combinations of Phone Number ===

def letter_combinations(digits):
    """Letter combinations from phone digits"""
    if not digits:
        return []
    
    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi',
        '5': 'jkl', '6': 'mno', '7': 'pqrs',
        '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, path):
        if index == len(digits):
            result.append(path)
            return
        
        for letter in mapping[digits[index]]:
            backtrack(index + 1, path + letter)
    
    backtrack(0, '')
    return result

# === Palindrome Partitioning ===

def partition_palindromes(s):
    """Partition string into palindromes"""
    result = []
    
    def is_palindrome(sub):
        return sub == sub[::-1]
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            if is_palindrome(s[start:end]):
                backtrack(end, path + [s[start:end]])
    
    backtrack(0, [])
    return result
```

---

## 📊 Summary Tables

### Greedy vs Divide & Conquer vs Backtracking

| Pattern | When to Use | Time Complexity | Example |
|---------|-------------|-----------------|---------|
| Greedy | Locally optimal leads to global | Varies | Activity selection |
| Divide & Conquer | Independent subproblems | O(n log n) | Merge sort |
| Backtracking | Explore all possibilities | Exponential | N-Queens |

### Common Patterns

| Pattern | Problem Type | Key Insight |
|---------|-------------|-------------|
| Interval scheduling | Overlapping intervals | Sort by end time |
| Two pointers | Sorted arrays | Move pointers based on condition |
| Sliding window | Subarrays/substrings | Expand/shrink window |
| Backtracking | Combinations/permutations | Try all possibilities |

---

## 🎯 ML Applications

| Algorithm Pattern | ML Application |
|------------------|----------------|
| Greedy | Decision trees, feature selection |
| Divide & Conquer | Parallel training |
| Backtracking | Hyperparameter search |
| Dynamic Programming | Sequence alignment, Viterbi |

---

## ❓ Quick Check Questions

1. What is the main difference between a Greedy algorithm and Dynamic Programming?
2. What are the three standard steps of a Divide and Conquer algorithm?
3. How does Backtracking differ from a brute-force approach?
4. In which scenario would Kadane's algorithm be chosen over a Divide and Conquer approach for finding the maximum subarray?
5. Why are Decision Trees in Machine Learning considered a Greedy algorithm?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. A **Greedy algorithm** makes the optimal choice at a local level, hoping it leads to a global optimum, and never reconsiders its choices. **Dynamic Programming** explores all possible paths (by solving subproblems) to guarantee a globally optimal solution.
2. The three steps are: **Divide** the problem into smaller subproblems, **Conquer** the subproblems by solving them recursively, and **Combine** the solutions of the subproblems to solve the original problem.
3. **Backtracking** is an optimization over brute-force. While brute-force blindly explores all possible paths to the very end, backtracking uses a bounding function to stop evaluating a path as soon as it determines the path cannot lead to a valid solution, effectively pruning the search tree.
4. **Kadane's algorithm** is chosen because it runs in $O(n)$ time with $O(1)$ space, making it significantly more efficient and easier to implement than the Divide and Conquer approach, which takes $O(n \log n)$ time and involves complex recursive logic.
5. **Decision Trees** are considered greedy because at each node, they pick the single feature split that provides the highest information gain *at that exact moment*, without looking ahead to see if a different split might lead to a better overall tree later on.

</details>
---

**Status:** ✅ Complete
**Next:** Practice Problems
