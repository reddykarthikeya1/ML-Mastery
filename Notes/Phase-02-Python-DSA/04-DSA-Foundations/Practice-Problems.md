# DSA Foundations - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Concept Recall
**1.1** What is the difference between $O(n)$ time complexity and $O(n)$ space complexity?
**1.2** Explain the difference between a Stack (LIFO) and a Queue (FIFO). Name one standard algorithm that relies on each.
**1.3** What is a Hash Collision, and what are two common ways to resolve it?
**1.4** What does it mean for a sorting algorithm to be "stable"?

### Level 2: Intermediate Operations
**2.1** **Two-Pointers:** Given a sorted array of integers, write the logic to find if any two numbers add up to a specific `target` in $O(n)$ time.
**2.2** **Linked Lists:** Describe the "Fast and Slow Pointer" (Tortoise and Hare) approach to finding the middle node of a singly linked list.
**2.3** **Trees:** Write the order of nodes visited in a Pre-order, In-order, and Post-order traversal of a simple Binary Tree with Root A, Left Child B, and Right Child C.
**2.4** **Sorting:** Why is Quick Sort preferred over Merge Sort for arrays in practice, even though its worst-case time complexity is $O(n^2)$ while Merge Sort is always $O(n \log n)$?

### Level 3: Advanced Data Structures & Algorithms
**3.1** **Dynamic Programming:** Identify the overlapping subproblems and optimal substructure in the classic "Coin Change" problem.
**3.2** **Graphs:** Explain how to detect a cycle in a Directed Graph using DFS and a "recursion stack" set.
**3.3** **Trie:** How does a Trie optimize the process of finding all words that start with a specific prefix compared to searching through a standard list of strings?

### Level 4: Python Implementation Practice
**4.1** Implement a basic `Stack` class using Python's `collections.deque`. Include `push`, `pop`, and `peek` methods.
**4.2** Implement the iterative version of Binary Search to find a `target` in a sorted list `nums`. Return the index, or `-1` if not found.
**4.3** Implement the classic DP solution for the Fibonacci sequence using Tabulation with $O(1)$ space optimization.

### Level 5: Advanced Algorithmic Design & Integration
**5.1** **Scenario:** You are designing the core engine for a "Real-time Network Monitoring" tool.
- **The Data:** A massive stream of IP addresses and connection latencies.
- **Requirements:**
    1. **Frequency:** Find the Top 100 most frequent IP addresses at any moment.
    2. **Shortest Path:** Find the fastest route between two servers in a dynamic weighted graph of 10,000+ nodes.
    3. **Efficiency:** These operations must run in near real-time.
**Task:** Describe the specific data structures and algorithms you would combine to solve this. 
- Which structure would you use for $O(1)$ frequency tracking vs. $O(\log k)$ Top-K retrieval?
- Which algorithm would you use for the shortest path, and how would you optimize it if the graph is sparse?
- How would you handle the "memory limit" if the number of unique IPs exceeds available RAM? (Hint: Think Bloom Filters or Count-Min Sketch).

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.2
A Stack returns the most recently added item first (Last-In-First-Out), heavily used in Depth-First Search (DFS). A Queue returns the oldest item first (First-In-First-Out), heavily used in Breadth-First Search (BFS).

### 2.1
```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current = arr[left] + arr[right]
        if current == target: return [left, right]
        elif current < target: left += 1
        else: right -= 1
    return [-1, -1]
```

### 2.3
- **Pre-order (Root, Left, Right):** A, B, C
- **In-order (Left, Root, Right):** B, A, C
- **Post-order (Left, Right, Root):** B, C, A

### 3.2
In a Directed Graph, a cycle exists if you encounter a node during DFS that is currently in your active recursion stack (meaning you are still exploring its descendants and have looped back to it). You maintain a `visited` set to avoid redundant work, and a separate `rec_stack` set to track the current path.

### 4.2
```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### 4.3
```python
def fib_optimized(n):
    if n <= 1: return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    return prev1
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23