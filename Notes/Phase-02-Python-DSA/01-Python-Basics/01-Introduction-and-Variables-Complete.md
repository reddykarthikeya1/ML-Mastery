# 2.1 Python Basics

## 🎯 Quick Overview
- **Programming fundamentals**: Variables, data types, operators
- **Control flow**: Conditionals and loops
- **Foundation for**: All Python programming

---

## 1. Introduction to Programming

### What is Programming?

```
Programming = Writing instructions for computers to execute

Compiled languages: C, C++, Java (convert to machine code before running)
Interpreted languages: Python, JavaScript (execute line by line)
```

### Python History

```
- Created by Guido van Rossum (1991)
- Python 3.x is current version (Python 2 deprecated in 2020)
- Known for: Readability, simplicity, extensive libraries
```

### Setting Up Python

```bash
# Check Python version
python --version

# Run Python interactively
python

# Run a script
python script.py
```

### Your First Program

```python
# print() function
print("Hello, World!")

# Comments
# This is a single-line comment

"""
This is a
multi-line comment (docstring)
"""
```

---

## 2. Variables and Data Types

### Variables and Assignment

```python
# Dynamic typing - no need to declare type
x = 5           # integer
y = 3.14        # float
name = "John"   # string
is_active = True  # boolean
nothing = None    # NoneType

# Type checking
print(type(x))      # <class 'int'>
print(isinstance(y, float))  # True
```

### Basic Data Types

```python
# int - integers
age = 25
negative = -10
big_num = 10**100  # Python handles arbitrarily large integers

# float - decimal numbers
pi = 3.14159
scientific = 1.5e-10  # 1.5 × 10^-10

# str - strings
name = "Python"
multiline = """Line 1
Line 2"""

# bool - boolean
is_true = True
is_false = False

# None - represents absence of value
result = None
```

### Type Conversion

```python
# Explicit conversion (casting)
int("5")        # 5
float("3.14")   # 3.14
str(42)         # "42"
bool(1)         # True
bool("")        # False

# Check type
type(5)         # <class 'int'>
```

### f-strings (String Formatting)

```python
name = "Alice"
age = 30

# f-strings (Python 3.6+)
print(f"Hello, {name}! You are {age} years old.")

# Format with precision
pi = 3.14159
print(f"Pi to 2 decimals: {pi:.2f}")  # 3.14

# Expressions in f-strings
print(f"Next year you'll be {age + 1}")
```

---

## 3. Operators

### Arithmetic Operators

```python
a, b = 10, 3

+   # Addition: a + b = 13
-   # Subtraction: a - b = 7
*   # Multiplication: a * b = 30
/   # Division: a / b = 3.333... (always float)
//  # Floor division: a // b = 3
%   # Modulus: a % b = 1
**  # Exponentiation: a ** b = 1000
```

### Comparison Operators

```python
==  # Equal to
!=  # Not equal to
<   # Less than
>   # Greater than
<=  # Less than or equal
>=  # Greater than or equal

# Chain comparisons
1 < x < 10  # Valid in Python!
```

### Logical Operators

```python
and  # Logical AND
or   # Logical OR
not  # Logical NOT

# Examples
True and False   # False
True or False    # True
not True         # False

# Short-circuit evaluation
x = False or "default"  # "default"
y = True and "value"    # "value"
```

### Bitwise Operators

```python
a, b = 5, 3  # 5 = 101, 3 = 011 in binary

&   # AND: a & b = 1 (001)
|   # OR: a | b = 7 (111)
^   # XOR: a ^ b = 6 (110)
~   # NOT: ~a = -6
<<  # Left shift: a << 1 = 10 (1010)
>>  # Right shift: a >> 1 = 2 (010)
```

### Membership and Identity Operators

```python
# Membership
x = [1, 2, 3]
1 in x      # True
4 not in x  # True

# Identity (checks if same object)
a = [1, 2]
b = [1, 2]
c = a

a == b      # True (same value)
a is b      # False (different objects)
a is c      # True (same object)
```

---

## 4. Control Structures - Conditionals

### if Statements

```python
age = 18

if age < 13:
    print("Child")
elif age < 20:
    print("Teenager")
else:
    print("Adult")
```

### Ternary Operator

```python
# Conditional expression
age = 20
status = "adult" if age >= 18 else "minor"

# Equivalent to:
if age >= 18:
    status = "adult"
else:
    status = "minor"
```

### Truthy and Falsy Values

```python
# Falsy values (evaluate to False)
False
None
0
""  # empty string
[]  # empty list
{}  # empty dict
()  # empty tuple

# Everything else is Truthy
if [1, 2]:  # True - non-empty list
    print("Has elements")
```

---

## 5. Control Structures - Loops

### for Loops

```python
# Iterate over sequence
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

# Iterate over string
for char in "hello":
    print(char)

# Iterate over list
for item in [1, 2, 3]:
    print(item)
```

### while Loops

```python
count = 0
while count < 5:
    print(count)
    count += 1
```

### Loop Control

```python
# break - exit loop
for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4

# continue - skip to next iteration
for i in range(5):
    if i == 2:
        continue
    print(i)  # 0, 1, 3, 4

# pass - do nothing (placeholder)
for i in range(5):
    pass  # TODO: implement later
```

### enumerate() and zip()

```python
# enumerate - get index and value
for i, val in enumerate(['a', 'b', 'c']):
    print(f"{i}: {val}")

# zip - iterate over multiple sequences
names = ['Alice', 'Bob']
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age}")
```

---

## 💻 Python Code Examples

```python
# === Example 1: Temperature Converter ===

def convert_temperature():
    """Convert between Celsius and Fahrenheit"""
    
    temp = float(input("Enter temperature: "))
    unit = input("Enter unit (C/F): ").upper()
    
    if unit == 'C':
        fahrenheit = (temp * 9/5) + 32
        print(f"{temp}°C = {fahrenheit}°F")
    elif unit == 'F':
        celsius = (temp - 32) * 5/9
        print(f"{temp}°F = {celsius}°C")
    else:
        print("Invalid unit")

# === Example 2: Number Guessing Game ===

import random

def guess_number():
    """Number guessing game"""
    
    secret = random.randint(1, 100)
    attempts = 0
    
    print("Guess the number (1-100):")
    
    while True:
        guess = int(input("Your guess: "))
        attempts += 1
        
        if guess < secret:
            print("Too low!")
        elif guess > secret:
            print("Too high!")
        else:
            print(f"Correct! It took {attempts} attempts")
            break

# === Example 3: Multiplication Table ===

def multiplication_table(n):
    """Print multiplication table for n"""
    
    for i in range(1, 11):
        print(f"{n} × {i} = {n * i}")

# Test
multiplication_table(7)
```

---

## 📊 Summary Tables

### Data Types

| Type | Example | Mutable | Use Case |
|------|---------|---------|----------|
| int | 42 | No | Whole numbers |
| float | 3.14 | No | Decimal numbers |
| str | "hello" | No | Text |
| bool | True/False | No | Boolean logic |
| None | None | No | Absence of value |

### Operators

| Category | Operators | Example |
|----------|-----------|---------|
| Arithmetic | +, -, *, /, //, %, ** | 10 // 3 = 3 |
| Comparison | ==, !=, <, >, <=, >= | 5 > 3 = True |
| Logical | and, or, not | True and False = False |
| Bitwise | &, \|, ^, ~, <<, >> | 5 & 3 = 1 |
| Membership | in, not in | 1 in [1,2,3] = True |
| Identity | is, is not | a is b |

---

## 🎯 ML Applications

| Python Concept | ML Application |
|---------------|----------------|
| Variables | Store model parameters |
| Conditionals | Decision boundaries |
| Loops | Training iterations |
| Operators | Feature calculations |

---

## ❓ Quick Check Questions

1. What's the difference between = and ==?
2. What does // do vs /?
3. How do you check the type of a variable?
4. What's the output of `bool("")`?
5. How do you write a multi-line comment?
6. What does `range(5)` produce?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **= vs ==:**
   - = is assignment
   - == is equality comparison

2. **// vs /:**
   - // is floor division (integer result)
   - / is regular division (float result)

3. **Check type:**
   - type(variable)

4. **bool("") output:**
   - False (empty string is falsy)

5. **Multi-line comment:**
   - Use triple quotes: """comment"""

6. **range(5):**
   - Produces: 0, 1, 2, 3, 4

</details>
---

**Status:** ✅ Complete
**Next:** Data Structures (Lists, Tuples, Sets, Dictionaries)
