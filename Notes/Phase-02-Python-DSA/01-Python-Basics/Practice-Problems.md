# Python Basics - Practice Problems

## Topic 1: Python Fundamentals

### Level 1: Basic

**1.1** Write a program that:
- Takes user input for name and age
- Prints a greeting with the year they'll turn 100

**1.2** Calculate:
- Area of circle given radius
- Fahrenheit to Celsius conversion
- Simple interest given principal, rate, time

**1.3** String manipulation:
- Reverse a string
- Count vowels in a string
- Check if string is palindrome

---

### Level 2: Intermediate

**2.1** List operations:
```python
# Given list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 1. Extract even numbers
# 2. Square all numbers
# 3. Get sum of all numbers
# 4. Find maximum without using max()
```

**2.2** Dictionary operations:
```python
# Create a program that:
# 1. Stores student names and grades
# 2. Calculates average grade
# 3. Finds student with highest grade
# 4. Updates a student's grade
```

**2.3** Python Practice - Functions:
```python
def fibonacci(n):
    """Return first n Fibonacci numbers"""
    # Your code here
    pass

def is_prime(n):
    """Check if number is prime"""
    # Your code here
    pass

# Test your functions
```

---

### Level 3: Advanced

**3.1** Class implementation:
```python
class BankAccount:
    """
    Implement a bank account class with:
    - deposit(amount)
    - withdraw(amount)
    - get_balance()
    - transaction_history()
    - Interest calculation (1% monthly)
    """
    pass
```

**3.2** File processing:
```python
"""
Read a CSV file with columns: name, age, salary
1. Calculate average salary
2. Find oldest and youngest person
3. Write results to output.txt
4. Handle file not found errors
"""
```

**3.3** Decorator challenge:
```python
def log_calls(func):
    """
    Create a decorator that logs:
    - Function name
    - Arguments passed
    - Return value
    - Execution time
    """
    pass

@log_calls
def slow_function():
    import time
    time.sleep(1)
    return "Done"
```

---

## Topic 2: Data Structures

### Level 1: Basic

**2.1** List comprehension:
```python
# Create using list comprehension:
# 1. Squares of 1-20
# 2. Even numbers from 1-100
# 3. All vowels in a given string
# 4. Matrix multiplication result
```

**2.2** Dictionary practice:
```python
# Word frequency counter
text = "hello world hello python world"
# Output: {'hello': 2, 'world': 2, 'python': 1}
```

---

### Level 2: Intermediate

**2.3** Implement stack using list:
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        pass
    
    def pop(self):
        pass
    
    def peek(self):
        pass
    
    def is_empty(self):
        pass
```

**2.4** Implement queue using collections.deque:
```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        pass
    
    def dequeue(self):
        pass
    
    def front(self):
        pass
```

---

## Topic 3: OOP

### Level 2: Intermediate

**3.1** Inheritance hierarchy:
```python
# Create class hierarchy:
# Animal (base)
#   ├── Mammal
#   │     ├── Dog
#   │     └── Cat
#   └── Bird
#       └── Eagle

# Each class should have:
# - __init__ method
# - speak() method (override)
# - Additional unique methods
```

**3.2** Polymorphism:
```python
# Create shapes: Circle, Rectangle, Triangle
# Each with:
# - area()
# - perimeter()
# Store in list and calculate total area
```

---

## Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
```python
name = input("Enter name: ")
age = int(input("Enter age: "))
year_100 = 2024 + (100 - age)
print(f"{name}, you'll turn 100 in {year_100}")
```

### 1.3
```python
# Reverse string
s = "hello"
reversed_s = s[::-1]

# Count vowels
vowels = sum(1 for c in s.lower() if c in 'aeiou')

# Palindrome check
is_palindrome = s == s[::-1]
```

### 2.1
```python
nums = list(range(1, 11))
evens = [x for x in nums if x % 2 == 0]
squares = [x**2 for x in nums]
total = sum(nums)
maximum = nums[0]
for n in nums[1:]:
    if n > maximum:
        maximum = n
```

### 3.1 (Stack)
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
