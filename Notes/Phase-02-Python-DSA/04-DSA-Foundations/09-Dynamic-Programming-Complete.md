# 5.9 Dynamic Programming

## 🎯 Quick Overview
- **DP**: Break problems into overlapping subproblems
- **Memoization**: Top-down with caching
- **Tabulation**: Bottom-up with table
- **Foundation for**: Optimization problems, sequence analysis, ML algorithms

---

## 1. DP Fundamentals

### When to Use DP

```
1. Overlapping subproblems
2. Optimal substructure
3. Can be solved recursively

Approaches:
- Memoization (Top-down): Cache recursive results
- Tabulation (Bottom-up): Build table iteratively
```

### Fibonacci Example

```python
# Naive recursion - O(2^n)
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)

# Memoization - O(n)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Tabulation - O(n)
def fib_tab(n):
    if n <= 1:
        return n
    
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[i-1] + dp[i-2])
    
    return dp[n]

# Space optimized - O(n) time, O(1) space
def fib_optimized(n):
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

---

## 2. Classic DP Problems

### Climbing Stairs

```python
def climb_stairs(n):
    """Number of ways to climb n stairs (1 or 2 at a time)"""
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Space optimized
def climb_stairs_opt(n):
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

### Coin Change

```python
def coin_change(coins, amount):
    """Minimum coins needed for amount"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def coin_change_ways(coins, amount):
    """Number of ways to make amount"""
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    
    return dp[amount]
```

### Longest Common Subsequence

```python
def lcs(text1, text2):
    """Length of longest common subsequence"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def lcs_with_string(text1, text2):
    """LCS with reconstruction"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct LCS
    lcs_str = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs_str.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs_str))
```

### Longest Increasing Subsequence

```python
def lis(nums):
    """Length of longest increasing subsequence - O(n²)"""
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def lis_optimized(nums):
    """LIS optimized - O(n log n)"""
    import bisect
    
    tails = []
    
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num
    
    return len(tails)
```

### 0/1 Knapsack

```python
def knapsack(weights, values, capacity):
    """0/1 Knapsack - maximum value"""
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

def knapsack_optimized(weights, values, capacity):
    """Space optimized knapsack"""
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

### Edit Distance

```python
def min_distance(word1, word2):
    """Minimum edit distance (Levenshtein distance)"""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    
    return dp[m][n]
```

---

## 3. DP Patterns

### 1D DP

```python
def house_robber(nums):
    """Rob houses without alerting police"""
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)
    
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[-1]

def decode_ways(s):
    """Number of ways to decode string"""
    if not s or s[0] == '0':
        return 0
    
    dp = [0] * (len(s) + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, len(s) + 1):
        # Single digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        
        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    
    return dp[len(s)]
```

### 2D DP

```python
def unique_paths(m, n):
    """Number of unique paths in grid"""
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]

def min_path_sum(grid):
    """Minimum path sum in grid"""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    dp[0][0] = grid[0][0]
    
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[m-1][n-1]
```

### Interval DP

```python
def matrix_chain_multiplication(dims):
    """Minimum scalar multiplications"""
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] + 
                       dims[i] * dims[k+1] * dims[j+1])
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n-1]
```

### DP with Bitmask

```python
def count_arrangements(n):
    """Count beautiful arrangements using bitmask DP"""
    memo = {}
    
    def backtrack(pos, used_mask):
        if pos > n:
            return 1
        
        if (pos, used_mask) in memo:
            return memo[(pos, used_mask)]
        
        count = 0
        for num in range(1, n + 1):
            if not (used_mask & (1 << num)) and (num % pos == 0 or pos % num == 0):
                count += backtrack(pos + 1, used_mask | (1 << num))
        
        memo[(pos, used_mask)] = count
        return count
    
    return backtrack(1, 0)
```

---

## 4. Advanced DP Problems

### Regular Expression Matching

```python
def is_match(s, p):
    """Regular expression matching with '.' and '*'"""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, a*b*c*
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                dp[i][j] = dp[i][j-2]  # Zero occurrence
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]  # One or more
    
    return dp[m][n]
```

### Burst Balloons

```python
def max_coins(nums):
    """Maximum coins from bursting balloons"""
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            for i in range(left + 1, right):
                coins = (nums[left] * nums[i] * nums[right] + 
                        dp[left][i] + dp[i][right])
                dp[left][right] = max(dp[left][right], coins)
    
    return dp[0][n-1]
```

### Distinct Subsequences

```python
def num_distinct(s, t):
    """Number of distinct subsequences of t in s"""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]
    
    return dp[m][n]
```

---

## 💻 Python Code Examples

```python
# === Word Break ===

def word_break(s, word_dict):
    """Check if string can be segmented"""
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[len(s)]

# === Partition Equal Subset Sum ===

def can_partition(nums):
    """Check if array can be partitioned into equal sum subsets"""
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    
    return dp[target]

# === Maximum Product Subarray ===

def max_product(nums):
    """Maximum product subarray"""
    if not nums:
        return 0
    
    max_prod = min_prod = result = nums[0]
    
    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        
        result = max(result, max_prod)
    
    return result
```

---

## 📊 Summary Tables

### DP Patterns

| Pattern | Problem Type | Example |
|---------|-------------|---------|
| 1D DP | Linear sequences | Fibonacci, Climbing stairs |
| 2D DP | Grids, strings | LCS, Edit distance |
| Interval DP | Ranges | Matrix chain multiplication |
| Bitmask DP | Subsets | Traveling salesman |
| Tree DP | Trees | Tree diameter |

### Common DP Problems

| Problem | Time | Space | Pattern |
|---------|------|-------|---------|
| Fibonacci | O(n) | O(1) | 1D DP |
| Coin Change | O(amount × coins) | O(amount) | Unbounded knapsack |
| LCS | O(mn) | O(mn) | 2D DP |
| 0/1 Knapsack | O(nW) | O(W) | 2D DP |
| Edit Distance | O(mn) | O(mn) | 2D DP |

---

## 🎯 ML Applications

| DP Concept | ML Application |
|------------|----------------|
| Sequence alignment | Bioinformatics, NLP |
| Viterbi algorithm | HMM decoding |
| Dynamic time warping | Time series analysis |
| Bellman-Ford | Reinforcement learning |

---

## ❓ Quick Check Questions

1. What are the two essential properties a problem must have to be solvable using Dynamic Programming?
2. What is the difference between Memoization and Tabulation?
3. In the 0/1 Knapsack problem, what does "0/1" mean?
4. How can you optimize the space complexity of the tabulation approach for the Fibonacci sequence?
5. Which DP pattern is typically used to find the Longest Common Subsequence of two strings?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. The problem must have **Overlapping Subproblems** (the same subproblems are solved repeatedly) and **Optimal Substructure** (the optimal solution to the main problem can be constructed from optimal solutions of its subproblems).
2. **Memoization** is a Top-Down approach that uses recursion and caches the results of function calls. **Tabulation** is a Bottom-Up approach that uses iteration to fill a table starting from the base cases up to the final solution.
3. "0/1" means that for every item, you must either entirely include it in the knapsack (1) or entirely exclude it (0). You cannot take fractions of an item.
4. Instead of storing the entire array of size $n$, you only need to store the last two calculated values (`prev1` and `prev2`) to calculate the current value, reducing the space complexity from $O(n)$ to **$O(1)$**.
5. The **2D DP** pattern is typically used. You create a 2D matrix where the rows represent characters of the first string and columns represent characters of the second string.

</details>
---

**Status:** ✅ Complete
**Next:** Algorithm Design Patterns
