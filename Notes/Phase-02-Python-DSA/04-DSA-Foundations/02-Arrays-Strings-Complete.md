# 5.2 Arrays and Strings

## 🎯 Quick Overview
- **Arrays**: Contiguous memory storage
- **Strings**: Character arrays
- **Two-pointer**: Efficient array traversal
- **Sliding window**: Subarray/substring problems
- **Foundation for**: Data manipulation, algorithm optimization

---

## 1. Array Operations

### Basic Operations

```python
# Creation
arr = [1, 2, 3, 4, 5]
arr2 = list(range(10))
arr3 = [0] * 5  # [0, 0, 0, 0, 0]

# Access
first = arr[0]
last = arr[-1]
middle = arr[2]

# Slicing
subarray = arr[1:4]  # [2, 3, 4]
reversed_arr = arr[::-1]

# Modification
arr[0] = 10
arr[1:3] = [20, 30]

# Methods
arr.append(6)
arr.insert(0, 0)
arr.remove(3)
arr.pop()  # Remove last
arr.pop(0)  # Remove first
```

### Array Problems

```python
def two_sum(nums, target):
    """Find two numbers that add up to target"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return [-1, -1]

def max_subarray(nums):
    """Kadane's algorithm - maximum sum subarray"""
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

def product_except_self(nums):
    """Product of array except self"""
    n = len(nums)
    result = [1] * n
    
    # Left products
    left = 1
    for i in range(n):
        result[i] = left
        left *= nums[i]
    
    # Right products
    right = 1
    for i in range(n-1, -1, -1):
        result[i] *= right
        right *= nums[i]
    
    return result
```

---

## 2. Two-Pointer Technique

### Pattern

```python
# Opposite ends
left, right = 0, len(arr) - 1
while left < right:
    # Process
    left += 1
    right -= 1

# Same direction
slow = fast = 0
while fast < len(arr):
    # Process
    fast += 1
    if condition:
        slow += 1
```

### Examples

```python
def two_sum_sorted(arr, target):
    """Two sum for sorted array"""
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

def remove_duplicates(nums):
    """Remove duplicates in-place (sorted array)"""
    if not nums:
        return 0
    
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1

def trap_rain_water(height):
    """Trapping rain water"""
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water
```

---

## 3. Sliding Window

### Pattern

```python
# Fixed size window
window_sum = sum(arr[:k])
max_sum = window_sum
for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i-k]
    max_sum = max(max_sum, window_sum)

# Variable size window
left = 0
for right in range(len(arr)):
    # Add arr[right] to window
    while condition_not_met:
        # Remove arr[left] from window
        left += 1
    # Update result
```

### Examples

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

def min_window_substring(s, t):
    """Minimum window substring containing all chars of t"""
    from collections import Counter
    
    need = Counter(t)
    missing = len(t)
    left = start = 0
    min_length = float('inf')
    
    for right, char in enumerate(s, 1):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        
        if missing == 0:
            while left < right and need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            
            if right - left < min_length:
                min_length = right - left
                start = left
            
            need[s[left]] += 1
            missing += 1
            left += 1
    
    return s[start:start + min_length] if min_length != float('inf') else ""
```

---

## 4. String Manipulation

### String Methods

```python
s = "  Hello, World!  "

# Case
s.lower()
s.upper()
s.title()
s.capitalize()
s.swapcase()

# Strip
s.strip()
s.lstrip()
s.rstrip()

# Search
s.find("World")
s.count("l")
s.startswith("He")
s.endswith("!")

# Replace and split
s.replace("World", "Python")
s.split(",")
"-".join(["Hello", "World"])

# Format
f"Hello, {s.strip()}!"
"{} {}".format("Hello", "World")
```

### String Problems

```python
def is_palindrome(s):
    """Check if string is palindrome"""
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

def valid_anagram(s1, s2):
    """Check if two strings are anagrams"""
    from collections import Counter
    return Counter(s1) == Counter(s2)

def longest_palindrome(s):
    """Longest palindromic substring"""
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    if not s:
        return ""
    
    start = end = 0
    for i in range(len(s)):
        len1 = expand(i, i)
        len2 = expand(i, i + 1)
        max_len = max(len1, len2)
        
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end + 1]

def group_anagrams(strs):
    """Group anagrams together"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())
```

---

## 5. Matrix Operations

### Matrix Traversal

```python
def spiral_order(matrix):
    """Spiral traversal of matrix"""
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Traverse down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        if top <= bottom:
            # Traverse left
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        if left <= right:
            # Traverse up
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result

def rotate_matrix(matrix):
    """Rotate matrix 90 degrees clockwise"""
    n = len(matrix)
    
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for i in range(n):
        matrix[i].reverse()
    
    return matrix
```

---

## 💻 Python Code Examples

```python
# === Container With Most Water ===

def max_area(height):
    """Container with most water"""
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        area = width * h
        max_area = max(max_area, area)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

# === 3Sum ===

def three_sum(nums):
    """Find all triplets that sum to zero"""
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
    
    return result

# === Merge Intervals ===

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

---

## 📊 Summary Tables

### Two-Pointer Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| Opposite ends | Sorted array search | Two sum |
| Same direction | Remove duplicates | Fast/slow pointers |
| Independent pointers | Merge intervals | Merge sorted arrays |

### Sliding Window Problems

| Problem Type | Window Type | Example |
|-------------|-------------|---------|
| Fixed size | Constant k | Max sum subarray |
| Variable size | Expand/shrink | Longest unique substring |
| Constrained | Meet condition | Minimum window substring |

---

## 🎯 ML Applications

| Array/String Concept | ML Application |
|---------------------|----------------|
| Array operations | Feature vectors |
| String manipulation | Text preprocessing |
| Sliding window | Sequence analysis |
| Matrix operations | Image processing |

---

## ❓ Quick Check Questions

1. When should you use the Two-Pointer technique starting from opposite ends?
2. What is the main advantage of the Sliding Window technique?
3. How do you reverse a string in-place in Python using slicing?
4. What is Kadane's algorithm primarily used for?
5. How does a fixed-size sliding window differ from a variable-size window?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. The opposite-ends Two-Pointer technique is typically used when dealing with **sorted arrays** (e.g., finding two numbers that sum to a target) or when checking for symmetry (e.g., palindromes).
2. The Sliding Window technique converts nested loops (which would have an $O(n^2)$ time complexity) into a single loop, reducing the time complexity to **$O(n)$** by keeping track of a subset of elements.
3. You can reverse a string using the slice notation: `s[::-1]`.
4. Kadane's algorithm is an $O(n)$ dynamic programming approach used to find the **maximum sum contiguous subarray** within a one-dimensional array of numbers.
5. A **fixed-size window** always maintains a constant width (e.g., exactly $k$ elements) as it moves through the array. A **variable-size window** expands or shrinks its left and right boundaries dynamically based on certain conditions (e.g., finding the longest substring without repeating characters).

</details>
---

**Status:** ✅ Complete
**Next:** Linked Lists
