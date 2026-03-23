# 2.6 File Handling and Error Handling

## 🎯 Quick Overview
- **File operations**: Read, write, append files
- **Exception handling**: Handle errors gracefully
- **Context managers**: Resource management
- **Foundation for**: Robust, production-ready code

---

## 1. File Operations

### Opening Files

```python
# Basic file operations
file = open('example.txt', 'r')  # 'r' = read, 'w' = write, 'a' = append
content = file.read()
file.close()

# Context manager (recommended)
with open('example.txt', 'r') as file:
    content = file.read()
# File automatically closed
```

### File Modes

```python
'r'   # Read (default)
'w'   # Write (overwrites or creates)
'a'   # Append
'x'   # Create (fails if exists)
'b'   # Binary mode
't'   # Text mode (default)
'+'   # Read and write
```

### Reading Files

```python
with open('file.txt', 'r') as f:
    content = f.read()      # Read entire file
    lines = f.readlines()   # Read all lines as list
    line = f.readline()     # Read one line
```

### Writing Files

```python
with open('file.txt', 'w') as f:
    f.write("Hello, World!\n")
    f.writelines(["Line 1\n", "Line 2\n"])
```

### Working with CSV and JSON

```python
import csv
import json

# CSV
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# JSON
data = {'name': 'Alice', 'age': 25}
with open('data.json', 'w') as f:
    json.dump(data, f)

with open('data.json', 'r') as f:
    data = json.load(f)
```

### pathlib for Modern Path Handling

```python
from pathlib import Path

# Create path
path = Path('data/file.txt')

# Check existence
path.exists()

# Read/write
content = path.read_text()
path.write_text('Hello')

# Path operations
path.parent      # Parent directory
path.name        # File name
path.suffix      # Extension
path.stem        # Name without extension
```

---

## 2. Exception Handling

### try-except Blocks

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Can't divide by zero!")
except Exception as e:
    print(f"Error: {e}")
else:
    print("No errors occurred")
finally:
    print("Always executes")
```

### Raising Exceptions

```python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

# Custom exceptions
class MyError(Exception):
    pass

raise MyError("Something went wrong")
```

### Exception Hierarchy

```python
BaseException
 ├── SystemExit
 ├── KeyboardInterrupt
 └── Exception
      ├── ValueError
      ├── TypeError
      ├── FileNotFoundError
      ├── ZeroDivisionError
      └── ...
```

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")
logging.error("Error message")
logging.critical("Critical message")
```

---

## 💻 Python Code Examples

```python
# === Example 1: Safe File Reader ===

def read_file_safely(file_path):
    """Read file with error handling"""
    
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except PermissionError:
        print(f"Permission denied: {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# === Example 2: Data Validator ===

def process_user_data(data):
    """Process user data with validation"""
    
    try:
        name = data['name']
        age = int(data['age'])
        email = data['email']
        
        if not name:
            raise ValueError("Name is required")
        
        if age < 0 or age > 150:
            raise ValueError("Invalid age")
        
        if '@' not in email:
            raise ValueError("Invalid email")
        
        return {"status": "success", "data": data}
    
    except KeyError as e:
        return {"status": "error", "message": f"Missing field: {e}"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected: {e}"}

# === Example 3: Context Manager ===

from contextlib import contextmanager

@contextmanager
def timer(name="Operation"):
    """Time how long a block takes"""
    import time
    
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{name} took {end - start:.4f} seconds")

# Usage
with timer("Processing"):
    # Some operation
    sum(range(1000000))
```

---

## 📊 Summary Tables

### File Modes

| Mode | Description | Creates File |
|------|-------------|--------------|
| 'r' | Read only | No |
| 'w' | Write (overwrite) | Yes |
| 'a' | Append | Yes |
| 'x' | Create only | Yes |
| 'r+' | Read and write | No |

### Exception Types

| Exception | When Raised |
|-----------|-------------|
| ValueError | Wrong value type |
| TypeError | Wrong type operation |
| FileNotFoundError | File doesn't exist |
| KeyError | Dict key not found |
| IndexError | List index out of range |
| ZeroDivisionError | Division by zero |

---

## 🎯 ML Applications

| Concept | ML Application |
|---------|----------------|
| File I/O | Load datasets |
| JSON/CSV | Data import/export |
| Exception handling | Robust pipelines |
| Logging | Training monitoring |
| Context managers | Resource management |

---

**Status:** ✅ Complete
**Next:** Advanced Python Topics
