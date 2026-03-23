# 2.5 Functional Programming

## 🎯 Quick Overview
- **Pure functions**: No side effects
- **Iterators/Generators**: Lazy evaluation
- **Functional tools**: map, filter, reduce
- **Foundation for**: Efficient data processing

---

## 1. Functional Programming Basics

### Pure Functions

```python
# Pure function - same input = same output, no side effects
def add(a, b):
    return a + b

# Impure function - has side effects
total = 0
def add_impure(a):
    global total
    total += a
    return total
```

### First-Class Functions

```python
# Functions can be:
# 1. Assigned to variables
def greet(name):
    return f"Hello, {name}!"

say_hello = greet
print(say_hello("Alice"))

# 2. Passed as arguments
def apply(func, value):
    return func(value)

apply(greet, "Bob")

# 3. Returned from functions
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

times_3 = make_multiplier(3)
times_3(5)  # 15
```

---

## 2. Built-in Functional Tools

### map(), filter(), reduce()

```python
from functools import reduce

nums = [1, 2, 3, 4, 5]

# map - apply function to all elements
squares = list(map(lambda x: x**2, nums))
# [1, 4, 9, 16, 25]

# filter - keep elements where function returns True
evens = list(filter(lambda x: x % 2 == 0, nums))
# [2, 4]

# reduce - combine elements
total = reduce(lambda x, y: x + y, nums)
# 15
```

### all() and any()

```python
nums = [1, 2, 3, 4, 5]

all(x > 0 for x in nums)   # True
any(x > 3 for x in nums)   # True
all(x % 2 == 0 for x in nums)  # False
```

---

## 3. itertools Module

```python
from itertools import count, cycle, repeat, accumulate, chain, combinations, permutations

# Infinite iterators
for i in count(0, 2):  # 0, 2, 4, 6, ...
    if i > 10:
        break

for item in cycle(['A', 'B', 'C']):  # A, B, C, A, B, C, ...
    pass

# Finite iterators
list(accumulate([1, 2, 3, 4]))  # [1, 3, 6, 10]
list(chain([1, 2], [3, 4]))     # [1, 2, 3, 4]

# Combinatoric iterators
list(combinations([1, 2, 3], 2))  # [(1,2), (1,3), (2,3)]
list(permutations([1, 2, 3]))     # [(1,2,3), (1,3,2), (2,1,3), ...]
```

---

## 4. functools Module

```python
from functools import partial, lru_cache, wraps

# partial functions
def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
cube = partial(power, exp=3)

square(5)  # 25
cube(3)    # 27

# lru_cache - memoization
@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# wraps - preserve function metadata
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

---

## 5. Generators and Iterators

### Iterables vs Iterators

```python
# Iterable - can be iterated over
my_list = [1, 2, 3]

# Iterator - object that produces next value
iterator = iter(my_list)
next(iterator)  # 1
next(iterator)  # 2
```

### Generator Functions

```python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i += 1

for num in count_up_to(5):
    print(num)  # 1, 2, 3, 4, 5

# Generator expression
squares = (x**2 for x in range(10))
for square in squares:
    print(square)
```

### yield from

```python
def chain_generators(gen1, gen2):
    yield from gen1
    yield from gen2
```

---

## 💻 Python Code Examples

```python
# === Example: Pipeline with Functional Tools ===

def process_data(data):
    """Process data using functional pipeline"""
    
    # Square even numbers, sum result
    result = (
        data
        |> (lambda x: filter(lambda n: n % 2 == 0, x))
        |> (lambda x: map(lambda n: n**2, x))
        |> (lambda x: reduce(lambda a, b: a + b, x))
    )
    
    return result

# === Example: Generator for Large Files ===

def read_large_file(file_path):
    """Read large file line by line"""
    
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

# Usage
for line in read_large_file('large_file.txt'):
    process(line)
```

---

## 📊 Summary Tables

### Functional Tools

| Tool | Purpose | Example |
|------|---------|---------|
| map() | Transform all elements | map(lambda x: x*2, nums) |
| filter() | Keep matching elements | filter(lambda x: x>0, nums) |
| reduce() | Combine elements | reduce(lambda a,b: a+b, nums) |
| all() | Check all True | all(x>0 for x in nums) |
| any() | Check any True | any(x>0 for x in nums) |

### Iterators

| Type | Syntax | Use Case |
|------|--------|----------|
| Iterator | iter(), next() | Manual iteration |
| Generator | yield | Lazy evaluation |
| Generator expr | (x for x in ...) | Memory-efficient loops |

---

## 🎯 ML Applications

| Functional Concept | ML Application |
|-------------------|----------------|
| map/filter | Data preprocessing |
| Generators | Batch data loading |
| lru_cache | Memoization |
| partial | Partial function application |

---

**Status:** ✅ Complete
**Next:** File Handling and Error Handling
