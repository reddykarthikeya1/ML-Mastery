# 2.2 Data Structures

## 🎯 Quick Overview
- **Lists**: Ordered, mutable collections
- **Tuples**: Ordered, immutable collections
- **Sets**: Unordered collections of unique elements
- **Dictionaries**: Key-value pairs
- **Foundation for**: Data manipulation in all Python programs

---

## 1. Strings

### String Creation and Indexing

```python
s = "Hello, World!"

# Indexing (0-based)
s[0]     # 'H'
s[-1]    # '!' (negative = from end)
s[7:12]  # 'World' (slicing)
s[::2]   # 'Hlo ol!' (step)
s[::-1]  # '!dlroW ,olleH' (reverse)
```

### String Methods

```python
s = "  Hello, World!  "

# Case
s.lower()      # "  hello, world!  "
s.upper()      # "  HELLO, WORLD!  "
s.title()      # "  Hello, World!  "
s.capitalize() # "  Hello, world!  "
s.swapcase()   # "  hELLO, wORLD!  "

# Strip whitespace
s.strip()      # "Hello, World!"
s.lstrip()     # "Hello, World!  "
s.rstrip()     # "  Hello, World!"

# Search
s.find("World")    # 7 (index)
s.count("l")       # 3
s.startswith("He") # True
s.endswith("!")    # True

# Replace and split
s.replace("World", "Python")  # "  Hello, Python!  "
s.split(",")       # ["  Hello", " World!  "]
s.split()          # ["Hello", "World!"] (split on whitespace)

# Join
"-".join(["a", "b", "c"])  # "a-b-c"
```

---

## 2. Lists

### List Creation

```python
# Create lists
nums = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
nested = [[1, 2], [3, 4]]
empty = []

# From other iterables
list("hello")  # ['h', 'e', 'l', 'l', 'o']
list(range(5)) # [0, 1, 2, 3, 4]
```

### List Operations

```python
nums = [1, 2, 3]

# Concatenation
nums + [4, 5]  # [1, 2, 3, 4, 5]

# Repetition
nums * 2  # [1, 2, 3, 1, 2, 3]

# Membership
2 in nums      # True
10 not in nums # True

# Length
len(nums)  # 3
```

### List Methods

```python
nums = [1, 2, 3]

# Add elements
nums.append(4)        # [1, 2, 3, 4]
nums.insert(1, 1.5)   # [1, 1.5, 2, 3, 4]
nums.extend([5, 6])   # [1, 1.5, 2, 3, 4, 5, 6]

# Remove elements
nums.remove(2)    # Remove first occurrence of 2
nums.pop()        # Remove and return last element
nums.pop(0)       # Remove and return element at index 0
nums.clear()      # Remove all elements

# Query
nums.index(3)     # Find index of 3
nums.count(2)     # Count occurrences of 2

# Sort and reverse
nums.sort()           # Sort in place
nums.sort(reverse=True)  # Sort descending
sorted(nums)          # Return new sorted list
nums.reverse()        # Reverse in place
```

### List Slicing

```python
nums = [0, 1, 2, 3, 4, 5]

nums[0:3]    # [0, 1, 2]
nums[::2]    # [0, 2, 4]
nums[::-1]   # [5, 4, 3, 2, 1, 0]
nums[1:-1]   # [1, 2, 3, 4]
```

### List Comprehensions

```python
# Basic comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]

# With transformation
words = ["hello", "world"]
upper = [w.upper() for w in words]
```

### Shallow vs Deep Copy

```python
import copy

# Shallow copy (references to same objects)
list1 = [1, 2, [3, 4]]
list2 = list1.copy()
list2[2][0] = 999
print(list1)  # [1, 2, [999, 4]] - affected!

# Deep copy (completely independent)
list3 = copy.deepcopy(list1)
list3[2][0] = 111
print(list1)  # [1, 2, [999, 4]] - unchanged!
```

---

## 3. Tuples

### Tuple Creation

```python
# Create tuples
t1 = (1, 2, 3)
t2 = 1, 2, 3  # Parentheses optional
single = (5,)  # Comma required for single element
empty = ()

# From other iterables
tuple("hello")  # ('h', 'e', 'l', 'l', 'o')
```

### Tuple Operations

```python
t = (1, 2, 3)

# Indexing and slicing (same as lists)
t[0]      # 1
t[1:3]    # (2, 3)

# Immutable - cannot modify!
# t[0] = 5  # TypeError!

# Methods (limited compared to lists)
t.count(2)    # 1
t.index(2)    # 1
```

### Tuple Packing and Unpacking

```python
# Packing
point = 3, 4, 5

# Unpacking
x, y, z = point

# Swap values
a, b = b, a

# Extended unpacking
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5
```

### Named Tuples

```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)

print(p.x)  # 3
print(p.y)  # 4
print(p)    # Point(x=3, y=4)
```

---

## 4. Sets

### Set Creation

```python
# Create sets
s1 = {1, 2, 3, 3, 2}  # {1, 2, 3} - duplicates removed
s2 = set([1, 2, 3])
empty = set()  # {} creates empty dict, not set!
```

### Set Operations

```python
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

# Union
A | B  # {1, 2, 3, 4, 5, 6}
A.union(B)

# Intersection
A & B  # {3, 4}
A.intersection(B)

# Difference
A - B  # {1, 2}
A.difference(B)

# Symmetric difference
A ^ B  # {1, 2, 5, 6}
A.symmetric_difference(B)

# Subset/Superset
{1, 2} <= A  # True (subset)
A >= {1, 2}  # True (superset)
```

### Set Methods

```python
s = {1, 2, 3}

# Add/remove
s.add(4)           # {1, 2, 3, 4}
s.remove(2)        # {1, 3, 4} - raises error if not found
s.discard(5)       # No error if not found
s.pop()            # Remove and return arbitrary element
s.clear()          # Empty the set

# Set comprehensions
squares = {x**2 for x in range(5)}  # {0, 1, 4, 9, 16}
```

### Frozen Sets

```python
# Immutable sets
fs = frozenset([1, 2, 3])
# fs.add(4)  # AttributeError - immutable!

# Can be used as dict keys
d = {fs: "value"}
```

---

## 5. Dictionaries

### Dictionary Creation

```python
# Create dictionaries
d1 = {'name': 'Alice', 'age': 25}
d2 = dict(name='Bob', age=30)
d3 = dict([('name', 'Charlie'), ('age', 35)])
empty = {}
```

### Dictionary Operations

```python
person = {'name': 'Alice', 'age': 25}

# Access
person['name']     # 'Alice'
person.get('age')  # 25
person.get('job', 'Unknown')  # 'Unknown' - default

# Modify
person['age'] = 26      # Update
person['job'] = 'Dev'   # Add new key

# Delete
del person['job']
person.pop('age')
person.popitem()  # Remove last inserted item
```

### Dictionary Methods

```python
d = {'a': 1, 'b': 2}

# Access
d.keys()    # dict_keys(['a', 'b'])
d.values()  # dict_values([1, 2])
d.items()   # dict_items([('a', 1), ('b', 2)])

# Update
d.update({'c': 3})  # {'a': 1, 'b': 2, 'c': 3}

# Copy
d.copy()  # Shallow copy
```

### Dictionary Comprehensions

```python
# Basic comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# With condition
evens = {x: x**2 for x in range(10) if x % 2 == 0}

# Swap keys and values
d = {'a': 1, 'b': 2}
swapped = {v: k for k, v in d.items()}
```

### Specialized Dictionaries

```python
from collections import OrderedDict, defaultdict, Counter

# OrderedDict - maintains insertion order
od = OrderedDict()
od['a'] = 1
od['b'] = 2

# defaultdict - default value for missing keys
dd = defaultdict(int)
dd['count'] += 1  # No error, defaults to 0

# Counter - count occurrences
c = Counter(['a', 'b', 'a', 'c', 'a'])
# Counter({'a': 3, 'b': 1, 'c': 1})
```

---

## 💻 Python Code Examples

```python
# === Example 1: List Operations ===

def list_practice():
    """Practice common list operations"""
    
    nums = list(range(1, 11))
    
    # Filter even numbers
    evens = [x for x in nums if x % 2 == 0]
    
    # Square all numbers
    squares = [x**2 for x in nums]
    
    # Sum all numbers
    total = sum(nums)
    
    # Find max without max()
    maximum = nums[0]
    for n in nums[1:]:
        if n > maximum:
            maximum = n
    
    print(f"Evens: {evens}")
    print(f"Squares: {squares}")
    print(f"Total: {total}")
    print(f"Max: {maximum}")

# === Example 2: Dictionary Practice ===

def analyze_text(text):
    """Count word frequency"""
    
    words = text.lower().split()
    frequency = {}
    
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    
    return frequency

text = "hello world hello python world"
freq = analyze_text(text)
print(freq)  # {'hello': 2, 'world': 2, 'python': 1}

# === Example 3: Set Operations ===

def find_common_and_unique():
    """Find common and unique elements between lists"""
    
    list1 = [1, 2, 3, 4, 5]
    list2 = [4, 5, 6, 7, 8]
    
    set1 = set(list1)
    set2 = set(list2)
    
    common = set1 & set2  # Intersection
    unique_to_1 = set1 - set2  # Difference
    unique_to_2 = set2 - set1
    all_unique = set1 | set2  # Union
    
    print(f"Common: {common}")
    print(f"Unique to list1: {unique_to_1}")
    print(f"Unique to list2: {unique_to_2}")
    print(f"All unique: {all_unique}")

# === Example 4: Nested Data Structures ===

def nested_structures():
    """Work with nested data structures"""
    
    # List of dictionaries
    students = [
        {'name': 'Alice', 'grades': [85, 90, 88]},
        {'name': 'Bob', 'grades': [75, 80, 82]},
        {'name': 'Charlie', 'grades': [95, 92, 98]}
    ]
    
    # Calculate average for each student
    for student in students:
        avg = sum(student['grades']) / len(student['grades'])
        print(f"{student['name']}: {avg:.1f}")
    
    # Find student with highest average
    best = max(students, key=lambda s: sum(s['grades'])/len(s['grades']))
    print(f"Best student: {best['name']}")

nested_structures()
```

---

## 📊 Summary Tables

### Data Structures Comparison

| Type | Ordered | Mutable | Unique | Syntax |
|------|---------|---------|--------|--------|
| list | Yes | Yes | No | [1, 2, 3] |
| tuple | Yes | No | No | (1, 2, 3) |
| set | No | Yes | Yes | {1, 2, 3} |
| dict | Yes* | Yes | Keys | {'a': 1} |

*Python 3.7+ dicts maintain insertion order

### Common Methods

| Structure | Common Methods |
|-----------|---------------|
| list | append(), extend(), insert(), remove(), pop(), sort() |
| tuple | count(), index() |
| set | add(), remove(), union(), intersection(), difference() |
| dict | keys(), values(), items(), get(), update(), pop() |

---

## 🎯 ML Applications

| Data Structure | ML Application |
|---------------|----------------|
| list | Feature vectors, batches |
| tuple | Data points (X, y) |
| set | Unique classes, vocabulary |
| dict | Hyperparameters, mappings |

---

## ❓ Quick Check Questions

1. What's the difference between list and tuple?
2. How do you remove duplicates from a list?
3. What happens if you access a non-existent dict key?
4. How do you create an empty set?
5. What does .get() do for dictionaries?
6. How do you swap two variables in one line?

---

## 📝 Answers to Quick Check

1. **list vs tuple:**
   - list is mutable, tuple is immutable
   - list uses [], tuple uses ()

2. **Remove duplicates:**
   - list(set(my_list))

3. **Non-existent dict key:**
   - Raises KeyError
   - Use .get() to avoid error

4. **Empty set:**
   - set() (not {} which is empty dict)

5. **.get() for dicts:**
   - Returns value or default if key doesn't exist

6. **Swap variables:**
   - a, b = b, a

---

**Status:** ✅ Complete
**Next:** Functions and Modules
