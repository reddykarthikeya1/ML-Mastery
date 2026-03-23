# 5.1 Complexity Analysis

## 🎯 Quick Overview
- **Time Complexity**: How runtime grows with input size
- **Space Complexity**: How memory usage grows with input size
- **Big O**: Upper bound (worst case)
- **Foundation for**: Algorithm selection, performance optimization

---

## 1. Time Complexity

### Big O Notation

```
Big O describes how runtime grows as input size increases

Common complexities (best to worst):
O(1)         - Constant
O(log n)     - Logarithmic
O(n)         - Linear
O(n log n)   - Linearithmic
O(n²)        - Quadratic
O(n³)        - Cubic
O(2ⁿ)        - Exponential
O(n!)        - Factorial
```

### Examples

```python
# O(1) - Constant time
def get_first(arr):
    return arr[0]

# O(log n) - Logarithmic (binary search)
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

# O(n) - Linear
def find_max(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

# O(n log n) - Linearithmic (merge sort)
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# O(n²) - Quadratic (bubble sort)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# O(2ⁿ) - Exponential (recursive Fibonacci)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# O(n!) - Factorial (permutations)
def permutations(arr):
    if len(arr) <= 1:
        return [arr]
    result = []
    for i in range(len(arr)):
        rest = permutations(arr[:i] + arr[i+1:])
        for perm in rest:
            result.append([arr[i]] + perm)
    return result
```

---

## 2. Space Complexity

### Definition

```
Space complexity = Auxiliary space + Input space

Auxiliary space: Extra space used by algorithm
Input space: Space for input data
```

### Examples

```python
# O(1) - Constant space
def sum_array(arr):
    total = 0
    for num in arr:
        total += num
    return total

# O(n) - Linear space
def copy_array(arr):
    copy = []
    for num in arr:
        copy.append(num)
    return copy

# O(n) - Recursive space (call stack)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
# Space: O(n) for call stack
```

---

## 3. Omega and Theta Notation

```
Big O (O): Upper bound (worst case)
Omega (Ω): Lower bound (best case)
Theta (Θ): Tight bound (average case)

Example: Linear search
- Best case: Ω(1) - found at first position
- Worst case: O(n) - found at last position or not found
- Average case: Θ(n) - on average search half the array
```

---

## 4. Analyzing Algorithms

### Rules

```
1. Drop constants: O(2n) → O(n)
2. Drop non-dominant terms: O(n² + n) → O(n²)
3. Add for sequential operations: O(a) + O(b) = O(a + b)
4. Multiply for nested operations: O(a × b)
```

### Examples

```python
# O(n + m)
def print_both_arrays(arr1, arr2):
    for item in arr1:
        print(item)
    for item in arr2:
        print(item)

# O(n²)
def print_pairs(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            print(arr[i], arr[j])

# O(n log n)
def process_and_sort(arr):
    # O(n)
    for item in arr:
        process(item)
    # O(n log n)
    return sorted(arr)
    # Total: O(n log n)
```

---

## 5. Amortized Analysis

```
Amortized time: Average time per operation over many operations

Example: Dynamic array append
- Most appends: O(1)
- Occasional resize: O(n)
- Amortized: O(1) per operation
```

---

## 📊 Summary Tables

### Complexity Comparison

| Complexity | Name | Example |
|------------|------|---------|
| O(1) | Constant | Array access |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Find max |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Bubble sort |
| O(2ⁿ) | Exponential | Recursive Fibonacci |
| O(n!) | Factorial | Permutations |

### Space Complexity

| Algorithm | Space | Reason |
|-----------|-------|--------|
| Iterative sum | O(1) | Single variable |
| Array copy | O(n) | New array |
| Merge sort | O(n) | Temporary arrays |
| Quick sort | O(log n) | Recursion stack |
| DFS | O(V) | Recursion stack |

---

## 🎯 ML Applications

| Complexity Concept | ML Application |
|-------------------|----------------|
| Time complexity | Algorithm selection |
| Space complexity | Memory optimization |
| Amortized analysis | Batch processing |

---

## ❓ Quick Check Questions

1. What is the difference between Big O and Omega (Ω) notation?
2. Why is an O(n log n) algorithm generally preferred over an O(n²) algorithm for large datasets?
3. What is the space complexity of a recursive algorithm that calls itself $n$ times?
4. How do you calculate the total time complexity of two sequential loops where the first runs $n$ times and the second runs $m$ times?
5. What does "Amortized O(1) time" mean?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Big O** represents the upper bound (worst-case scenario) of an algorithm's growth rate, while **Omega (Ω)** represents the lower bound (best-case scenario).
2. As $n$ (input size) grows very large, $n^2$ grows much faster than $n \log n$. Therefore, an $O(n \log n)$ algorithm (like Merge Sort) will execute significantly faster than an $O(n^2)$ algorithm (like Bubble Sort) on large datasets.
3. The space complexity is **O(n)** because each recursive call adds a new frame to the call stack in memory.
4. The total time complexity is **O(n + m)**. Because the loops are sequential, their complexities are added.
5. **Amortized O(1) time** means that while a single operation might occasionally be expensive (e.g., O(n) to resize a dynamic array), when averaged out over a sequence of many operations, the cost per operation is constant O(1).

</details>
---

**Status:** ✅ Complete
**Next:** Arrays and Strings
