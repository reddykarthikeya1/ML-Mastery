# 2.7 Advanced Python Topics

## 🎯 Quick Overview
- **Decorators**: Modify function behavior
- **Context managers**: Resource management
- **Type hints**: Type annotations
- **Concurrency**: Threading, multiprocessing, asyncio
- **Testing**: Unit tests, pytest
- **Foundation for**: Professional Python development

---

## 1. Decorators

### Function Decorators

```python
def timer(func):
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
    time.sleep(1)

slow_function()
```

### Decorators with Arguments

```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
```

### Built-in Decorators

```python
class MyClass:
    @staticmethod
    def static_method():
        pass
    
    @classmethod
    def class_method(cls):
        pass
    
    @property
    def my_property(self):
        pass
```

---

## 2. Context Managers

### Using with Statement

```python
# Built-in context managers
with open('file.txt', 'r') as f:
    content = f.read()

with lock:  # Threading lock
    # Critical section
    pass
```

### Creating Context Managers

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)

with managed_resource() as r:
    use_resource(r)
```

### Class-based Context Managers

```python
class ManagedResource:
    def __enter__(self):
        self.resource = acquire()
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        release(self.resource)

with ManagedResource() as r:
    use(r)
```

---

## 3. Type Hints

### Basic Type Annotations

```python
from typing import List, Dict, Optional, Union, Tuple

def greet(name: str) -> str:
    return f"Hello, {name}!"

def process_items(items: List[int]) -> Dict[str, int]:
    return {"count": len(items)}

def get_value() -> Optional[str]:
    return None  # or return "value"

def process(value: Union[int, str]) -> str:
    return str(value)

def get_point() -> Tuple[int, int]:
    return (0, 0)
```

### Type Checking with mypy

```bash
# Install mypy
pip install mypy

# Run type checking
mypy script.py
```

---

## 4. Concurrency Basics

### Threading

```python
import threading

def worker():
    print("Working...")

thread = threading.Thread(target=worker)
thread.start()
thread.join()
```

### Multiprocessing

```python
from multiprocessing import Process

def worker():
    print("Working in separate process")

p = Process(target=worker)
p.start()
p.join()
```

### asyncio (Async/Await)

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "Data"

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

### GIL (Global Interpreter Lock)

```
GIL prevents multiple threads from executing Python bytecode simultaneously

Impact:
- CPU-bound tasks: Use multiprocessing
- I/O-bound tasks: Use threading or asyncio
```

---

## 5. Testing

### unittest

```python
import unittest

def add(a, b):
    return a + b

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_add_negative(self):
        self.assertEqual(add(-1, -1), -2)

if __name__ == '__main__':
    unittest.main()
```

### pytest

```python
# test_math.py
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

# Run: pytest test_math.py
```

### Test Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15
```

### Mocking

```python
from unittest.mock import patch

@patch('module.function_to_mock')
def test_with_mock(mock_func):
    mock_func.return_value = 42
    # Test code
```

---

## 💻 Python Code Examples

```python
# === Example 1: Complete Decorator Suite ===

from functools import wraps
import time

def log_calls(func):
    """Log function calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

def cache_result(func):
    """Cache function results"""
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@log_calls
@cache_result
def expensive_operation(x, y):
    time.sleep(1)
    return x + y

# === Example 2: Async Web Scraper ===

import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = ['http://example.com', 'http://example.org']
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        for url, content in zip(urls, results):
            print(f"{url}: {len(content)} bytes")

# asyncio.run(main())

# === Example 3: Complete Testing Example ===

# calculator.py
class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# test_calculator.py
import pytest
from calculator import Calculator

@pytest.fixture
def calc():
    return Calculator()

def test_add(calc):
    assert calc.add(2, 3) == 5

def test_divide(calc):
    assert calc.divide(10, 2) == 5

def test_divide_by_zero(calc):
    with pytest.raises(ValueError):
        calc.divide(10, 0)
```

---

## 📊 Summary Tables

### Decorators

| Decorator | Purpose | Example |
|-----------|---------|---------|
| @staticmethod | Define static method | @staticmethod def func(): |
| @classmethod | Define class method | @classmethod def func(cls): |
| @property | Create property | @property def value(self): |
| @wraps(func) | Preserve metadata | @wraps(func) |

### Concurrency

| Approach | Best For | Module |
|----------|----------|--------|
| Threading | I/O-bound tasks | threading |
| Multiprocessing | CPU-bound tasks | multiprocessing |
| asyncio | Async I/O | asyncio |

### Testing

| Tool | Purpose | Command |
|------|---------|---------|
| unittest | Built-in testing | python -m unittest |
| pytest | Advanced testing | pytest |
| mock | Mocking objects | unittest.mock |
| coverage | Code coverage | coverage run |

---

## 🎯 ML Applications

| Advanced Topic | ML Application |
|---------------|----------------|
| Decorators | Timing, caching, logging |
| Type hints | Code quality, IDE support |
| Concurrency | Parallel data loading |
| Testing | Model testing, CI/CD |
| Context managers | Resource management |

---

## ❓ Quick Check Questions

1. What does @wraps do?
2. When to use threading vs multiprocessing?
3. What's the benefit of type hints?
4. How does async/await work?
5. What's a test fixture?
6. When to use context managers?

---

## 📝 Answers to Quick Check

1. **@wraps:**
   - Preserves original function metadata
   - Used in decorator definitions

2. **Threading vs Multiprocessing:**
   - Threading: I/O-bound (file, network)
   - Multiprocessing: CPU-bound (computation)

3. **Type hints benefits:**
   - Better IDE support
   - Catch type errors early
   - Self-documenting code

4. **async/await:**
   - async defines coroutine
   - await pauses until result ready
   - Allows concurrent execution

5. **Test fixture:**
   - Setup code run before tests
   - Provides test data/resources

6. **Context managers:**
   - Automatic resource cleanup
   - File handling, locks, connections

---

**Status:** ✅ Complete
**Next:** Python for Data Science (NumPy, Pandas, Visualization)
