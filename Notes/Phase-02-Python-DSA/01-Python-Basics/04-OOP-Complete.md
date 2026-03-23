# 2.4 Object-Oriented Programming

## 🎯 Quick Overview
- **Classes**: Blueprints for objects
- **Inheritance**: Code reuse
- **Polymorphism**: Multiple forms
- **Foundation for**: Organized, modular code

---

## 1. OOP Fundamentals

### Classes and Objects

```python
class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
    
    # Instance method
    def bark(self):
        return f"{self.name} says woof!"

# Create objects
dog1 = Dog("Buddy", 3)
dog2 = Dog("Milo", 5)

print(dog1.name)    # "Buddy"
print(dog1.bark())  # "Buddy says woof!"
```

### The `__init__` Constructor

```python
class Person:
    def __init__(self, name="Unknown", age=0):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"Hi, I'm {self.name}, {self.age} years old"

p1 = Person()  # Uses defaults
p2 = Person("Alice", 25)
```

### `__str__` and `__repr__`

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"Person({self.name}, {self.age})"
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

p = Person("Alice", 25)
print(str(p))   # "Person(Alice, 25)"
print(repr(p))  # "Person('Alice', 25)"
```

---

## 2. Encapsulation

### Public, Protected, Private

```python
class BankAccount:
    def __init__(self, balance):
        self.balance = balance      # Public
        self._account_type = "Savings"  # Protected (convention)
        self.__password = "1234"    # Private (name mangling)
    
    def deposit(self, amount):
        self.balance += amount
    
    def __get_balance(self):  # Private method
        return self.balance
```

### @property Decorator

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32

temp = Temperature(25)
print(temp.fahrenheit)  # 77.0
temp.celsius = 30       # Uses setter
```

---

## 3. Inheritance

### Basic Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"

class Dog(Animal):  # Dog inherits from Animal
    def speak(self):  # Override parent method
        return f"{self.name} says woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())  # "Buddy says woof!"
print(cat.speak())  # "Whiskers says meow!"
```

### super() Function

```python
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # Call parent constructor
        self.age = age
```

### Multiple Inheritance and MRO

```python
class A:
    def greet(self):
        return "Hello from A"

class B:
    def greet(self):
        return "Hello from B"

class C(A, B):  # Multiple inheritance
    pass

c = C()
print(c.greet())  # "Hello from A" (first in MRO)
print(C.__mro__)  # Method Resolution Order
```

### Abstract Base Classes

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # Error - can't instantiate abstract class
rect = Rectangle(5, 3)  # OK
```

---

## 4. Polymorphism

### Duck Typing

```python
class Duck:
    def speak(self):
        return "Quack!"

class Person:
    def speak(self):
        return "Hello!"

def make_speak(obj):
    print(obj.speak())

make_speak(Duck())   # "Quack!"
make_speak(Person()) # "Hello!"
```

### Operator Overloading

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(4, 5)
print(v1 + v2)  # "Vector(6, 8)"
print(v1 * 3)   # "Vector(6, 9)"
```

---

## 5. Advanced OOP

### Class Methods and Static Methods

```python
class MyClass:
    class_var = "I'm a class variable"
    
    def __init__(self, value):
        self.instance_var = value
    
    @classmethod
    def class_method(cls):
        return cls.class_var
    
    @staticmethod
    def static_method(x, y):
        return x + y
    
    def instance_method(self):
        return self.instance_var

print(MyClass.class_method())     # "I'm a class variable"
print(MyClass.static_method(2, 3)) # 5
```

### Composition vs Inheritance

```python
# Inheritance (is-a relationship)
class Animal:
    pass

class Dog(Animal):  # Dog IS-A Animal
    pass

# Composition (has-a relationship)
class Engine:
    def start(self):
        return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()  # Car HAS-A Engine
    
    def start(self):
        return self.engine.start()
```

---

## 💻 Python Code Examples

```python
# === Complete Example: Bank Account System ===

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private
        self.transactions = []
    
    @property
    def balance(self):
        return self.__balance
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            self.transactions.append(f"Deposited: ${amount}")
            return f"Deposited ${amount}"
        return "Invalid amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            self.transactions.append(f"Withdrew: ${amount}")
            return f"Withdrew ${amount}"
        return "Insufficient funds"
    
    def get_statement(self):
        return f"\n".join(self.transactions)
    
    def __str__(self):
        return f"{self.owner}'s Account: ${self.balance}"

# Usage
account = BankAccount("Alice", 1000)
print(account.deposit(500))
print(account.withdraw(200))
print(account.balance)
print(account.get_statement())
```

---

## 📊 Summary Tables

### OOP Concepts

| Concept | Keyword | Purpose |
|---------|---------|---------|
| Class | class | Define blueprint |
| Inheritance | class Child(Parent) | Reuse code |
| Encapsulation | @property | Control access |
| Polymorphism | Method override | Multiple forms |
| Abstraction | @abstractmethod | Hide complexity |

### Method Types

| Type | Decorator | First Parameter | Use Case |
|------|-----------|-----------------|----------|
| Instance | None | self | Object-specific behavior |
| Class | @classmethod | cls | Class-level operations |
| Static | @staticmethod | None | Utility functions |

---

## 🎯 ML Applications

| OOP Concept | ML Application |
|-------------|----------------|
| Classes | Custom estimators, models |
| Inheritance | Model hierarchies |
| Polymorphism | Unified interfaces |
| Encapsulation | Data protection |

---

---

## ❓ Quick Check Questions

1. What is the purpose of the `self` parameter in instance methods?
2. How does inheritance facilitate code reuse?
3. What is "name mangling" in Python, and how is it triggered?
4. What is the difference between `@classmethod` and `@staticmethod`?
5. When would you use composition over inheritance?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. The **`self` parameter** is a reference to the specific instance of the class. It allows instance methods to access and modify the attributes and other methods of that particular object.
2. **Inheritance** allows a child class to acquire all the attributes and methods of a parent class. This avoids duplicating code and allows the child class to extend or specialize the behavior of the parent.
3. **Name mangling** is Python's way of making a class attribute private. It is triggered by prefixing an attribute name with two underscores (e.g., `__balance`). Python internally renames it to `_ClassName__attribute` to prevent accidental access or modification from outside the class.
4. A **`@classmethod`** receives the class itself (`cls`) as its first argument and can access class-level attributes. A **`@staticmethod`** receives no special first argument and behaves like a regular function, but is grouped inside the class for logical organization.
5. Use **composition** (has-a relationship) when you want to build complex objects by combining simpler ones, providing more flexibility. Use **inheritance** (is-a relationship) when there is a clear hierarchical relationship and you want to reuse and extend the core logic of a base class.

</details>
---

**Status:** ✅ Complete
**Next:** Functional Programming
