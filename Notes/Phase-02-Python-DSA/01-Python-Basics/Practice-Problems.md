# Python Basics - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Syntax and Data Types
**1.1** Write a Python script that calculates the area of a circle given its radius. Use the `math` module for the value of $\pi$.
**1.2** Given the string `s = "Machine Learning"`, write the code to:
- Print only the word "Machine".
- Print the string in reverse.
- Count the number of occurrences of the letter 'e'.
**1.3** Explain the difference between a `list` and a `tuple` in terms of mutability and syntax.
**1.4** Create a dictionary representing a student with keys `name`, `age`, and `courses` (a list). Add a new course to the list.

### Level 2: Control Flow and Functions
**2.1** Write a function `is_prime(n)` that returns `True` if a number is prime and `False` otherwise.
**2.2** Use a list comprehension to create a list of squares for all even numbers between 1 and 20.
**2.3** Write a function `describe_person(name, **kwargs)` that prints the name and then iterates through the keyword arguments to print "Key: Value" pairs.
**2.4** Explain the difference between `global` and `nonlocal` keywords with a small code example.

### Level 3: Intermediate OOP and Functional Programming
**3.1** Create a class `BankAccount` with a private attribute `__balance`. Implement a `@property` for balance and methods for `deposit` and `withdraw` with basic validation (e.g., can't withdraw more than balance).
**3.2** Create a class hierarchy: `Shape` (base) -> `Rectangle` and `Circle` (derived). Implement an `area()` method in both using polymorphism.
**3.3** Use `map()` and `filter()` to take a list of integers, keep only the odd ones, and double them.
**3.4** Write a generator function `fibonacci_gen(n)` that yields the first $n$ Fibonacci numbers.

### Level 4: Advanced Python & File Handling
**4.1** Write a decorator `timer` that prints the execution time of any function it decorates.
**4.2** Write a script that reads a CSV file `data.csv`, calculates the average of a numeric column, and writes the result to a new file `output.txt` using a `with` statement.
**4.3** Use the `itertools` module to find all possible permutations of the list `[1, 2, 3]`.
**4.4** Implement a custom context manager using either a class (with `__enter__` and `__exit__`) or the `@contextmanager` decorator.

### Level 5: High-Concurrency & System Internals
**5.1** **Scenario:** You are building a high-performance Data Ingestion Engine for a real-time ML platform.
- **Goal:** Download 1,000 large JSON files from a remote API concurrently.
- **Constraints:** 
    1. You must use **`asyncio`** and `aiohttp` to maximize I/O efficiency.
    2. To prevent overwhelming the server, implement a **semaphore** to limit concurrency to 10 simultaneous downloads.
    3. To save memory, do not load all files into a single list; use a **generator** to yield processed records one-by-one to a downstream consumer.
    4. Implement a **custom decorator** `@retry` that automatically retries a download up to 3 times if a `TimeoutError` occurs.
**Task:** Implementation logic outline. Describe how you would combine these advanced Python features to build a robust, memory-efficient ingestion system. Explain how the **GIL** affects this specific I/O-bound task.

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 2.2
```python
squares = [x**2 for x in range(1, 21) if x % 2 == 0]
```

### 3.1
```python
class BankAccount:
    def __init__(self, balance=0):
        self.__balance = balance
    
    @property
    def balance(self):
        return self.__balance
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
```

### 4.1
```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time: {time.time() - start}")
        return result
    return wrapper
```

### 4.2
```python
import csv
try:
    with open('data.csv', 'r') as f:
        reader = csv.DictReader(f)
        vals = [float(row['score']) for row in reader]
        avg = sum(vals)/len(vals)
    with open('output.txt', 'w') as f:
        f.write(f"Average: {avg}")
except FileNotFoundError:
    print("File missing")
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
