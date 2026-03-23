# 5.8 Sorting and Searching

## 🎯 Quick Overview
- **Sorting**: Arrange elements in order
- **Searching**: Find element in collection
- **Complexity**: Time and space efficiency
- **Foundation for**: Data organization, efficient lookup

---

## 1. Sorting Algorithms

### Comparison Sorts

```python
def bubble_sort(arr):
    """O(n²) time, O(1) space, stable"""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

def selection_sort(arr):
    """O(n²) time, O(1) space, not stable"""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    """O(n²) time, O(1) space, stable"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

### Divide and Conquer

```python
def merge_sort(arr):
    """O(n log n) time, O(n) space, stable"""
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

def quick_sort(arr):
    """O(n log n) avg, O(n²) worst, O(log n) space, not stable"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr):
    """O(n log n) time, O(1) space, not stable"""
    import heapq
    
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]
```

### Non-Comparison Sorts

```python
def counting_sort(arr):
    """O(n+k) time, O(k) space, stable"""
    if not arr:
        return []
    
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1
    
    count = [0] * range_val
    output = [0] * len(arr)
    
    for num in arr:
        count[num - min_val] += 1
    
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    for num in reversed(arr):
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1
    
    return output

def radix_sort(arr):
    """O(d*(n+b)) time, O(n+b) space, stable"""
    if not arr:
        return []
    
    max_val = max(arr)
    exp = 1
    
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for num in arr:
        index = (num // exp) % 10
        count[index] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    for num in reversed(arr):
        index = (num // exp) % 10
        output[count[index] - 1] = num
        count[index] -= 1
    
    for i in range(n):
        arr[i] = output[i]
```

---

## 2. Searching Algorithms

### Linear Search

```python
def linear_search(arr, target):
    """O(n) time, O(1) space"""
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

def find_all_occurrences(arr, target):
    """Find all indices of target"""
    return [i for i, x in enumerate(arr) if x == target]
```

### Binary Search

```python
def binary_search(arr, target):
    """O(log n) time, O(1) space (iterative)"""
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

def binary_search_recursive(arr, target, left=0, right=None):
    """O(log n) time, O(log n) space (recursive)"""
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

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

def binary_search_rightmost(arr, target):
    """Find rightmost occurrence"""
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left - 1 if arr[left - 1] == target else -1
```

### Binary Search Variants

```python
def search_rotated_array(arr, target):
    """Search in rotated sorted array"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        
        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

def find_min_rotated(arr):
    """Find minimum in rotated sorted array"""
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid
    
    return arr[left]

def search_2d_matrix(matrix, target):
    """Search in 2D sorted matrix"""
    if not matrix:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_val = matrix[mid // cols][mid % cols]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False
```

---

## 3. Advanced Search Problems

```python
def find_peak_element(arr):
    """Find peak element (greater than neighbors)"""
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left

def search_insert_position(arr, target):
    """Find position where target should be inserted"""
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

def find_first_bad_version(n, is_bad_version):
    """Find first bad version"""
    left, right = 1, n
    
    while left < right:
        mid = (left + right) // 2
        
        if is_bad_version(mid):
            right = mid
        else:
            left = mid + 1
    
    return left

def sqrt(x):
    """Integer square root"""
    if x == 0:
        return 0
    
    left, right = 1, x
    
    while left <= right:
        mid = (left + right) // 2
        
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right
```

---

## 4. Sorting Problems

```python
def sort_colors(nums):
    """Sort colors (0=red, 1=white, 2=blue) - Dutch flag"""
    left, mid, right = 0, 0, len(nums) - 1
    
    while mid <= right:
        if nums[mid] == 0:
            nums[left], nums[mid] = nums[mid], nums[left]
            left += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[right] = nums[right], nums[mid]
            right -= 1

def merge_sorted_arrays(arr1, arr2):
    """Merge two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result

def kth_largest(arr, k):
    """Find kth largest element"""
    import heapq
    
    min_heap = arr[:k]
    heapq.heapify(min_heap)
    
    for num in arr[k:]:
        if num > min_heap[0]:
            heapq.heapreplace(min_heap, num)
    
    return min_heap[0]

def top_k_frequent(nums, k):
    """Top K frequent elements"""
    from collections import Counter
    
    count = Counter(nums)
    return [num for num, _ in count.most_common(k)]
```

---

## 💻 Python Code Examples

```python
# === Median of Two Sorted Arrays ===

def find_median_sorted_arrays(nums1, nums2):
    """Find median of two sorted arrays"""
    # Ensure nums1 is smaller
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    raise ValueError("Input arrays are not sorted")

# === H-Index ===

def h_index(citations):
    """Calculate h-index"""
    citations.sort(reverse=True)
    
    for i, citation in enumerate(citations):
        if citation < i + 1:
            return i
    
    return len(citations)

# === Meeting Rooms II ===

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
```

---

## 📊 Summary Tables

### Sorting Algorithms Comparison

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes |

### Searching Algorithms

| Algorithm | Time | Space | Requirement |
|-----------|------|-------|-------------|
| Linear Search | O(n) | O(1) | None |
| Binary Search | O(log n) | O(1) | Sorted array |
| Hash Search | O(1) avg | O(n) | Hash table |

---

## 🎯 ML Applications

| Sorting/Searching | ML Application |
|-------------------|----------------|
| Sorting | Data preprocessing |
| Binary Search | Hyperparameter tuning |
| Top-K | Recommendation systems |
| Kth element | Feature selection |

---

**Status:** ✅ Complete
**Next:** Dynamic Programming
