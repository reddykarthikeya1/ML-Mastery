# 2.3 Functions and Modules

## 🎯 Quick Overview
- **Functions**: Reusable blocks of code
- **Modules**: Organized code files
- **Foundation for**: Code organization and reusability

---

## 1. Functions

### Defining Functions

```python
def greet(name):
    """Return a greeting"""
    return f"Hello, {name}!"

# Call function
greet("Alice")  # "Hello, Alice!"
```

### Parameters and Arguments

```python
# Positional arguments
def add(a, b):
    return a + b

add(2, 3)  # 5

# Default parameters
def greet(name="World"):
    return f"Hello, {name}!"

greet()        # "Hello, World!"
greet("Alice") # "Hello, Alice!"

# Keyword arguments
def describe_pet(name, animal_type="dog"):
    print(f"{name} is a {animal_type}")

describe_pet(animal_type="cat", name="Whiskers")

# *args - variable positional arguments
def sum_all(*args):
    return sum(args)

sum_all(1, 2, 3, 4)  # 10

# **kwargs - variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25)
```

### Return Statement

```python
# Multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

min_val, max_val, avg = get_stats([1, 2, 3, 4, 5])

# No return = returns None
def print_hello():
    print("Hello")

result = print_hello()  # result = None
```

### Scope (LEGB Rule)

```python
# LEGB: Local, Enclosing, Global, Built-in

x = "global"

def outer():
    x = "enclosing"
    
    def inner():
        x = "local"
        print(x)
    
    inner()  # "local"

outer()

# global keyword
def modify_global():
    global x
    x = "modified"

# nonlocal keyword
def outer():
    x = "enclosing"
    
    def inner():
        nonlocal x
        x = "modified"
    
    inner()
    print(x)  # "modified"
```

### Lambda Functions

```python
# Anonymous functions
square = lambda x: x**2
square(5)  # 25

# Use with higher-order functions
nums = [1, 2, 3, 4]
list(map(lambda x: x**2, nums))  # [1, 4, 9, 16]
list(filter(lambda x: x % 2 == 0, nums))  # [2, 4]
```

---

## 2. Modules and Packages

### Importing Modules

```python
# Import entire module
import math
math.sqrt(16)  # 4.0

# Import specific functions
from math import sqrt, pi
sqrt(16)  # 4.0

# Import with alias
import numpy as np
import pandas as pd

# Import all (not recommended)
from math import *
```

### Creating Modules

```python
# mymodule.py
def greet(name):
    return f"Hello, {name}!"

PI = 3.14159

# main.py
import mymodule
mymodule.greet("Alice")
```

### The `if __name__ == "__main__"` Idiom

```python
# mymodule.py
def main():
    print("Running as script")

if __name__ == "__main__":
    main()
# Only runs when executed directly, not when imported
```

### Standard Library Overview

```python
# Common modules
import os           # Operating system interface
import sys          # System-specific parameters
import math         # Mathematical functions
import random       # Random number generation
import datetime     # Date and time
import json         # JSON encoding/decoding
import csv          # CSV file handling
import re           # Regular expressions
import collections  # Specialized container datatypes
import itertools    # Iterator functions
import functools    # Higher-order functions
```

### Installing Packages

```bash
# Using pip
pip install package_name

# requirements.txt
# numpy==1.21.0
# pandas==1.3.0
pip install -r requirements.txt

# Virtual environments
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## 💻 Python Code Examples

```python
# === Example 1: Function with Multiple Return Values ===

def analyze_numbers(numbers):
    """Return statistics of a list of numbers"""
    
    if not numbers:
        return None, None, None, None
    
    return {
        'min': min(numbers),
        'max': max(numbers),
        'mean': sum(numbers) / len(numbers),
        'count': len(numbers)
    }

result = analyze_numbers([1, 2, 3, 4, 5])
print(result)

# === Example 2: Decorator ===

def timer(func):
    """Time how long a function takes"""
    import time
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f}s")
        return result
    
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "Done"

# === Example 3: Module Creation ===

# Save as: mymath.py
"""
mymath.py - Custom math module
"""

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

PI = 3.14159

# Save as: main.py
"""
main.py - Using mymath module
"""
import mymath

print(mymath.factorial(5))  # 120
print(mymath.fibonacci(10))  # 55
print(mymath.PI)  # 3.14159
```

---

## 📊 Summary Tables

### Function Types

| Type | Syntax | Use Case |
|------|--------|----------|
| Regular | def func(): ... | General purpose |
| Lambda | lambda x: x+1 | Simple one-liners |
| Method | def method(self): ... | Inside classes |
| Generator | def gen(): yield x | Lazy evaluation |

### Import Methods

| Method | Syntax | When to Use |
|--------|--------|-------------|
| Full module | import math | When using many functions |
| Specific | from math import sqrt | When using few functions |
| Alias | import numpy as np | Long module names |
| All | from math import * | Never (pollutes namespace) |

---

## 🎯 ML Applications

| Function Concept | ML Application |
|-----------------|----------------|
| Functions | Model definitions |
| Lambda | Custom transformations |
| Modules | Code organization |
| Decorators | Timing, caching |

---

**Status:** ✅ Complete
**Next:** Object-Oriented Programming
