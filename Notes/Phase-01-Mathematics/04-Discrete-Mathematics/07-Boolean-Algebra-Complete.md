# 1.4.7 Boolean Algebra

## 🎯 Quick Overview
- **Boolean operations**: AND, OR, NOT
- **Boolean identities**: Simplification rules
- **Logic gates**: Digital circuit building blocks
- **Foundation for**: Digital circuits, programming, SQL

---

## 1. Boolean Operations

### Basic Operations

| Operation | Symbol | Meaning | Truth Table |
|-----------|--------|---------|-------------|
| **AND** | x · y or xy | x and y | 1 iff both 1 |
| **OR** | x + y | x or y | 0 iff both 0 |
| **NOT** | x' or ¬x or x̄ | not x | opposite |
| **XOR** | x ⊕ y | x xor y | 1 iff different |
| **NAND** | (xy)' | not AND | 0 iff both 1 |
| **NOR** | (x+y)' | not OR | 1 iff both 0 |

### Truth Tables

```
x | y | xy | x+y | x' | x⊕y | (xy)' | (x+y)'
--+---+----+-----+----+-----+-------+-------
0 | 0 |  0 |  0  |  1 |  0  |   1   |   1
0 | 1 |  0 |  1  |  1 |  1  |   1   |   0
1 | 0 |  0 |  1  |  0 |  1  |   1   |   0
1 | 1 |  1 |  1  |  0 |  0  |   0   |   0
```

---

## 2. Boolean Identities

### Basic Laws

| Law | Formula |
|-----|---------|
| **Identity** | x + 0 = x, x · 1 = x |
| **Null** | x + 1 = 1, x · 0 = 0 |
| **Idempotent** | x + x = x, x · x = x |
| **Complement** | x + x' = 1, x · x' = 0 |
| **Involution** | (x')' = x |

### Algebraic Laws

| Law | Formula |
|-----|---------|
| **Commutative** | x + y = y + x, xy = yx |
| **Associative** | (x + y) + z = x + (y + z) |
| **Distributive** | x(y + z) = xy + xz |
| **De Morgan's** | (x + y)' = x'y', (xy)' = x' + y' |
| **Absorption** | x + xy = x, x(x + y) = x |

---

## 3. Boolean Expressions

### Forms

**Sum of Products (SOP):**
```
F = xy + x'y' + yz

OR of AND terms
```

**Product of Sums (POS):**
```
F = (x + y)(x' + z)

AND of OR terms
```

### Simplification

```
Using identities to reduce expression

Example:
F = xy + xy'
  = x(y + y')     [Distributive]
  = x(1)          [Complement]
  = x             [Identity]
```

---

## 4. Logic Gates

### Basic Gates

| Gate | Symbol | Output |
|------|--------|--------|
| **AND** | D-shape | A · B |
| **OR** | Curved | A + B |
| **NOT** | Triangle + circle | A' |
| **NAND** | AND + circle | (A · B)' |
| **NOR** | OR + circle | (A + B)' |
| **XOR** | OR with extra line | A ⊕ B |
| **XNOR** | XOR + circle | (A ⊕ B)' |

### Universal Gates

```
NAND and NOR are universal:
Any Boolean function can be built using only NAND or only NOR
```

---

## 5. Karnaugh Maps (K-maps)

### Purpose

**Visual method for simplifying Boolean expressions**

### 2-Variable K-map

```
      y'   y
x' |  00 | 01 |
x  |  10 | 11 |

Group adjacent 1s in powers of 2 (1, 2, 4, 8...)
Each group gives a product term
```

### 3-Variable K-map

```
       y'z'  y'z  yz  yz'
x'  |  000 | 001 | 011 | 010 |
x   |  100 | 101 | 111 | 110 |

Wrap around edges (toroidal)
```

### Simplification Rules

```
1. Group all 1s
2. Make groups as large as possible
3. Each group must be power of 2
4. Minimize number of groups
5. Each 1 must be in at least one group
```

---

## 6. Digital Circuit Basics

### Combinational Circuits

```
Output depends only on current input

Examples:
- Adders
- Multiplexers
- Encoders/Decoders
```

### Sequential Circuits

```
Output depends on input AND previous state

Examples:
- Flip-flops
- Registers
- Memory
```

### Half Adder

```
Inputs: A, B
Outputs: Sum, Carry

Sum = A ⊕ B
Carry = A · B
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt

# === Boolean Operations ===

def AND(a, b):
    return a and b

def OR(a, b):
    return a or b

def NOT(a):
    return not a

def XOR(a, b):
    return a != b

def NAND(a, b):
    return not (a and b)

def NOR(a, b):
    return not (a or b)

# Generate truth table
print("Truth Table")
print("=" * 70)
print(f"{'A':<5} {'B':<5} {'A·B':<5} {'A+B':<5} {'A\'':<5} {'A⊕B':<5} {'(AB)\'':<5} {'(A+B)\'':<5}")
print("-" * 70)

for a in [0, 1]:
    for b in [0, 1]:
        print(f"{a:<5} {b:<5} {AND(a,b):<5} {OR(a,b):<5} {int(NOT(a)):<5} {XOR(a,b):<5} {int(NAND(a,b)):<5} {int(NOR(a,b)):<5}")

# === Boolean Expression Evaluator ===

def evaluate_expression(a, b, c):
    """Evaluate F = AB + A'C + BC"""
    return (a and b) or ((not a) and c) or (b and c)

print("\nBoolean Expression: F = AB + A'C + BC")
print("=" * 50)
print(f"{'A':<5} {'B':<5} {'C':<5} {'F':<5}")
print("-" * 30)

for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            f = evaluate_expression(a, b, c)
            print(f"{a:<5} {b:<5} {c:<5} {int(f):<5}")

# === Boolean Identity Verifier ===

def verify_demorgan():
    """Verify De Morgan's Laws"""
    print("\nDe Morgan's Laws Verification")
    print("=" * 50)
    
    print("(A + B)' = A'B'")
    print("(AB)' = A' + B'")
    print()
    
    for a in [0, 1]:
        for b in [0, 1]:
            left1 = int(NOT(OR(a, b)))
            right1 = int(AND(NOT(a), NOT(b)))
            
            left2 = int(NOT(AND(a, b)))
            right2 = int(OR(NOT(a), NOT(b)))
            
            print(f"A={a}, B={b}:")
            print(f"  (A+B)' = {left1}, A'B' = {right1}, Match: {left1==right1}")
            print(f"  (AB)' = {left2}, A'+B' = {right2}, Match: {left2==right2}")

verify_demorgan()

# === Logic Gate Simulator ===

class LogicGate:
    """Base class for logic gates"""
    
    def __init__(self, name):
        self.name = name
    
    def output(self, *inputs):
        raise NotImplementedError

class ANDGate(LogicGate):
    def output(self, *inputs):
        return all(inputs)

class ORGate(LogicGate):
    def output(self, *inputs):
        return any(inputs)

class NOTGate(LogicGate):
    def output(self, *inputs):
        return not inputs[0]

class XORGate(LogicGate):
    def output(self, *inputs):
        result = inputs[0]
        for inp in inputs[1:]:
            result = result != inp
        return result

# Test gates
print("\nLogic Gate Simulator")
print("=" * 40)

and_gate = ANDGate("AND")
or_gate = ORGate("OR")
not_gate = NOTGate("NOT")
xor_gate = XORGate("XOR")

print(f"AND(1, 0, 1) = {and_gate.output(1, 0, 1)}")
print(f"OR(0, 0, 1) = {or_gate.output(0, 0, 1)}")
print(f"NOT(1) = {not_gate.output(1)}")
print(f"XOR(1, 0, 1) = {xor_gate.output(1, 0, 1)}")

# === Circuit Builder ===

class Circuit:
    """Build simple circuits from gates"""
    
    def __init__(self):
        self.gates = []
    
    def half_adder(self, a, b):
        """Half adder circuit"""
        sum_out = XORGate("XOR").output(a, b)
        carry = ANDGate("AND").output(a, b)
        return sum_out, carry
    
    def full_adder(self, a, b, cin):
        """Full adder circuit"""
        sum1, carry1 = self.half_adder(a, b)
        sum_out, carry2 = self.half_adder(sum1, cin)
        carry_out = ORGate("OR").output(carry1, carry2)
        return sum_out, carry_out

# Test circuits
print("\nHalf Adder")
print("=" * 30)
print(f"{'A':<5} {'B':<5} {'Sum':<5} {'Carry':<5}")
print("-" * 30)

circuit = Circuit()
for a in [0, 1]:
    for b in [0, 1]:
        s, c = circuit.half_adder(a, b)
        print(f"{a:<5} {b:<5} {int(s):<5} {int(c):<5}")

print("\nFull Adder")
print("=" * 40)
print(f"{'A':<5} {'B':<5} {'Cin':<5} {'Sum':<5} {'Cout':<5}")
print("-" * 40)

for a in [0, 1]:
    for b in [0, 1]:
        for cin in [0, 1]:
            s, c = circuit.full_adder(a, b, cin)
            print(f"{a:<5} {b:<5} {int(cin):<5} {int(s):<5} {int(c):<5}")

# === K-map Visualizer ===

def draw_kmap_2var(minterms):
    """Draw 2-variable K-map"""
    
    print("\n2-Variable K-map")
    print("=" * 30)
    print("      y'   y")
    print("x' |", end="")
    for y in [0, 1]:
        val = 0 if (0, y) in minterms else 0
        print(f"  {1 if (0, y) in minterms else 0} ", end="")
    print("|")
    print("x  |", end="")
    for y in [0, 1]:
        print(f"  {1 if (1, y) in minterms else 0} ", end="")
    print("|")

# Example: F = Σ(1, 2, 3)
draw_kmap_2var([(0, 1), (1, 0), (1, 1)])

# === Boolean Function Simplifier ===

def simplify_boolean():
    """Demonstrate Boolean simplification"""
    
    print("\nBoolean Simplification Example")
    print("=" * 50)
    
    # Original: F = AB + AB'
    # Simplified: F = A
    
    print("Original: F = AB + AB'")
    print("\nSimplification steps:")
    print("1. F = AB + AB'")
    print("2. F = A(B + B')     [Distributive]")
    print("3. F = A(1)          [Complement]")
    print("4. F = A             [Identity]")
    
    # Verify with truth table
    print("\nVerification:")
    print(f"{'A':<5} {'B':<5} {'AB+AB\'':<10} {'A':<5} {'Match':<5}")
    print("-" * 45)
    
    for a in [0, 1]:
        for b in [0, 1]:
            original = (a and b) or (a and (not b))
            simplified = a
            match = "✓" if original == simplified else "✗"
            print(f"{a:<5} {b:<5} {int(original):<10} {simplified:<5} {match:<5}")

simplify_boolean()

# === Universal Gate Demonstration ===

def nand_only_not(a):
    """NOT using only NAND"""
    return NAND(a, a)

def nand_only_and(a, b):
    """AND using only NAND"""
    return NAND(NAND(a, b), NAND(a, b))

def nand_only_or(a, b):
    """OR using only NAND"""
    return NAND(NAND(a, a), NAND(b, b))

print("\nNAND is Universal")
print("=" * 40)

print("\nNOT from NAND:")
for a in [0, 1]:
    print(f"NOT({a}) = {nand_only_not(a)}")

print("\nAND from NAND:")
for a in [0, 1]:
    for b in [0, 1]:
        print(f"AND({a}, {b}) = {nand_only_and(a, b)}")

print("\nOR from NAND:")
for a in [0, 1]:
    for b in [0, 1]:
        print(f"OR({a}, {b}) = {nand_only_or(a, b)}")
```

---

## 📊 Summary Tables

### Boolean Operations

| Operation | Symbol | Truth Table |
|-----------|--------|-------------|
| AND | xy or x·y | 1 iff both 1 |
| OR | x + y | 0 iff both 0 |
| NOT | x' or x̄ | opposite |
| XOR | x ⊕ y | 1 iff different |

### Boolean Identities

| Identity | Formula |
|----------|---------|
| Identity | x + 0 = x, x·1 = x |
| Null | x + 1 = 1, x·0 = 0 |
| Complement | x + x' = 1, x·x' = 0 |
| De Morgan's | (x+y)' = x'y', (xy)' = x'+y' |
| Absorption | x + xy = x, x(x+y) = x |

### Logic Gates

| Gate | Output | Use |
|------|--------|-----|
| AND | A·B | Conjunction |
| OR | A+B | Disjunction |
| NOT | A' | Negation |
| NAND | (A·B)' | Universal |
| NOR | (A+B)' | Universal |
| XOR | A⊕B | Addition |

---

## 🎯 ML Applications

| Application | Boolean Algebra Concept |
|-------------|------------------------|
| **Decision Trees** | Boolean conditions |
| **Rule-Based Systems** | Logic gates |
| **Neural Networks** | Activation (on/off) |
| **Database Queries** | Boolean operators |
| **Feature Engineering** | Boolean features |

---

**Status:** ✅ Complete
**Next:** Practice Problems
