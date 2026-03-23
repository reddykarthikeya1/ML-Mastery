# 5.5 Hash Tables

## 🎯 Quick Overview
- **Hash Table**: Key-value storage with O(1) operations
- **Hash Function**: Maps keys to indices
- **Collision Resolution**: Handle same hash values
- **Foundation for**: Dictionaries, sets, caching, databases

---

## 1. Hash Table Implementation

### Basic Implementation

```python
class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]  # Chaining
    
    def _hash(self, key):
        """Hash function"""
        return hash(key) % self.size
    
    def put(self, key, value):
        """Insert key-value pair"""
        index = self._hash(key)
        
        # Check if key exists
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        
        # Add new key-value
        self.table[index].append((key, value))
    
    def get(self, key):
        """Get value for key"""
        index = self._hash(key)
        
        for k, v in self.table[index]:
            if k == key:
                return v
        
        raise KeyError(key)
    
    def remove(self, key):
        """Remove key-value pair"""
        index = self._hash(key)
        
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                return
        
        raise KeyError(key)
    
    def contains(self, key):
        """Check if key exists"""
        index = self._hash(key)
        
        for k, v in self.table[index]:
            if k == key:
                return True
        
        return False
```

### Open Addressing Implementation

```python
class HashTableOpenAddressing:
    def __init__(self, size=100):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def _probe(self, index):
        """Linear probing"""
        return (index + 1) % self.size
    
    def put(self, key, value):
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value
                return
            index = self._probe(index)
        
        self.keys[index] = key
        self.values[index] = value
    
    def get(self, key):
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = self._probe(index)
        
        raise KeyError(key)
```

---

## 2. Hash Table Problems

### Two Sum Problems

```python
def two_sum(nums, target):
    """Two sum - find indices of two numbers"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return [-1, -1]

def three_sum(nums):
    """Three sum - find all triplets that sum to zero"""
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

def four_sum(nums, target):
    """Four sum - find all quadruplets that sum to target"""
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j-1]:
                continue
            
            left, right = j + 1, n - 1
            while left < right:
                total = nums[i] + nums[j] + nums[left] + nums[right]
                
                if total < target:
                    left += 1
                elif total > target:
                    right -= 1
                else:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
    
    return result
```

### Grouping Problems

```python
def group_anagrams(strs):
    """Group anagrams together"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

def group_anagrams_optimized(strs):
    """Group anagrams using character count as key"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        groups[tuple(count)].append(s)
    
    return list(groups.values())

def find_duplicates(nums):
    """Find all duplicates in array"""
    from collections import Counter
    
    count = Counter(nums)
    return [num for num, freq in count.items() if freq > 1]

def contains_duplicate(nums):
    """Check if array contains any duplicates"""
    return len(nums) != len(set(nums))

def contains_nearby_duplicate(nums, k):
    """Check if duplicate exists within k distance"""
    seen = {}
    for i, num in enumerate(nums):
        if num in seen and i - seen[num] <= k:
            return True
        seen[num] = i
    return False
```

### Subarray Problems

```python
def subarray_sum(nums, k):
    """Subarray sum equals K"""
    from collections import defaultdict
    
    count = defaultdict(int)
    count[0] = 1
    current_sum = 0
    result = 0
    
    for num in nums:
        current_sum += num
        if current_sum - k in count:
            result += count[current_sum - k]
        count[current_sum] += 1
    
    return result

def longest_consecutive(nums):
    """Longest consecutive sequence"""
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        if num - 1 not in num_set:  # Start of sequence
            current_num = num
            current_length = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            max_length = max(max_length, current_length)
    
    return max_length

def max_subarray_len(nums, k):
    """Maximum size subarray sum equals k"""
    sum_to_index = {0: -1}
    current_sum = 0
    max_len = 0
    
    for i, num in enumerate(nums):
        current_sum += num
        if current_sum - k in sum_to_index:
            max_len = max(max_len, i - sum_to_index[current_sum - k])
        if current_sum not in sum_to_index:
            sum_to_index[current_sum] = i
    
    return max_len
```

---

## 3. Set Operations

### Set Implementation

```python
class MySet:
    def __init__(self):
        self.table = {}
    
    def add(self, value):
        self.table[value] = True
    
    def remove(self, value):
        if value in self.table:
            del self.table[value]
    
    def contains(self, value):
        return value in self.table
    
    def union(self, other):
        result = MySet()
        for val in self.table:
            result.add(val)
        for val in other.table:
            result.add(val)
        return result
    
    def intersection(self, other):
        result = MySet()
        for val in self.table:
            if other.contains(val):
                result.add(val)
        return result
    
    def difference(self, other):
        result = MySet()
        for val in self.table:
            if not other.contains(val):
                result.add(val)
        return result
```

### Set Problems

```python
def intersection(nums1, nums2):
    """Intersection of two arrays"""
    return list(set(nums1) & set(nums2))

def union(nums1, nums2):
    """Union of two arrays"""
    return list(set(nums1) | set(nums2))

def difference(nums1, nums2):
    """Difference of two arrays"""
    return list(set(nums1) - set(nums2))

def is_happy(n):
    """Happy number"""
    seen = set()
    
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))
    
    return n == 1
```

---

## 4. Advanced Hash Table Problems

### LRU Cache

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key not in self.cache:
            return -1
        
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)
```

### Word Pattern

```python
def word_pattern(pattern, s):
    """Word pattern matching"""
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    char_to_word = {}
    word_to_char = {}
    
    for char, word in zip(pattern, words):
        if char in char_to_word:
            if char_to_word[char] != word:
                return False
        else:
            if word in word_to_char:
                return False
            char_to_word[char] = word
            word_to_char[word] = char
    
    return True

def is_isomorphic(s, t):
    """Isomorphic strings"""
    if len(s) != len(t):
        return False
    
    s_to_t = {}
    t_to_s = {}
    
    for c1, c2 in zip(s, t):
        if c1 in s_to_t:
            if s_to_t[c1] != c2:
                return False
        else:
            if c2 in t_to_s:
                return False
            s_to_t[c1] = c2
            t_to_s[c2] = c1
    
    return True
```

---

## 💻 Python Code Examples

```python
# === First Unique Character ===

def first_unique_char(s):
    """First unique character in string"""
    from collections import Counter
    
    count = Counter(s)
    
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    
    return -1

# === Valid Sudoku ===

def is_valid_sudoku(board):
    """Valid sudoku checker"""
    seen = set()
    
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != '.':
                row_key = f"R{i}{num}"
                col_key = f"C{j}{num}"
                box_key = f"B{i//3}{j//3}{num}"
                
                if row_key in seen or col_key in seen or box_key in seen:
                    return False
                
                seen.add(row_key)
                seen.add(col_key)
                seen.add(box_key)
    
    return True

# === Minimum Window Substring ===

def min_window(s, t):
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

## 📊 Summary Tables

### Hash Table Operations

| Operation | Average | Worst | Space |
|-----------|---------|-------|-------|
| Search | O(1) | O(n) | O(n) |
| Insert | O(1) | O(n) | O(n) |
| Delete | O(1) | O(n) | O(n) |

### Collision Resolution

| Method | Pros | Cons |
|--------|------|------|
| Chaining | Simple, handles overflow | Extra space for pointers |
| Linear Probing | Cache-friendly | Clustering |
| Quadratic Probing | Less clustering | Secondary clustering |
| Double Hashing | Minimal clustering | Two hash functions |

---

## 🎯 ML Applications

| Hash Table Concept | ML Application |
|-------------------|----------------|
| Hashing | Feature hashing |
| Caching | Model caching |
| Set operations | Vocabulary building |
| Frequency counting | Text analysis |

---

**Status:** ✅ Complete
**Next:** Trees and BSTs
