# 1.4.4 Number Theory Basics

## 🎯 Quick Overview
- **Divisibility**: When one number divides another
- **Primes**: Building blocks of integers
- **GCD/LCM**: Common divisors and multiples
- **Modular arithmetic**: Clock arithmetic
- **Foundation for**: Cryptography, hashing, algorithms

---

## 1. Divisibility

### Definition

```
a divides b (written a | b) if b = ak for some integer k

a is divisor/factor of b
b is multiple of a
```

### Properties

```
1. If a | b and a | c, then a | (b + c)
2. If a | b and b | c, then a | c (transitive)
3. If a | b and b | a, then a = ±b
4. If a | b, then a | bc for any c
```

### Examples

```
3 | 12 because 12 = 3 × 4
5 ∤ 12 because 12/5 is not integer

Divisors of 12: 1, 2, 3, 4, 6, 12
```

---

## 2. Prime Numbers

### Definition

**Prime:** Integer > 1 with exactly two positive divisors: 1 and itself

```
Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, ...

Composite: Has divisors other than 1 and itself
1 is neither prime nor composite
```

### Fundamental Theorem of Arithmetic

```
Every integer n > 1 can be uniquely written as:

n = p₁^a₁ × p₂^a₂ × ... × pₖ^aₖ

where pᵢ are distinct primes, aᵢ ≥ 1

Example:
60 = 2² × 3¹ × 5¹
```

### Prime Testing

```
Trial division: Check divisibility up to √n

If n is composite, it has a factor ≤ √n
```

---

## 3. GCD and LCM

### Greatest Common Divisor

```
gcd(a, b) = largest integer dividing both a and b

Examples:
gcd(12, 18) = 6
gcd(17, 19) = 1 (coprime)
gcd(0, n) = n
```

### Least Common Multiple

```
lcm(a, b) = smallest positive integer divisible by both a and b

Examples:
lcm(4, 6) = 12
lcm(3, 5) = 15
```

### Relationship

```
gcd(a, b) × lcm(a, b) = |a × b|

lcm(a, b) = |a × b| / gcd(a, b)
```

---

## 4. Euclidean Algorithm

### Finding GCD

```
gcd(a, b) = gcd(b, a mod b)

Algorithm:
1. Divide a by b, get remainder r
2. Replace a with b, b with r
3. Repeat until r = 0
4. Answer is the last non-zero remainder
```

### Example

```
gcd(48, 18):
48 = 18 × 2 + 12
18 = 12 × 1 + 6
12 = 6 × 2 + 0

gcd(48, 18) = 6
```

### Extended Euclidean Algorithm

```
Also finds x, y such that:
ax + by = gcd(a, b)

Useful for modular inverse
```

---

## 5. Modular Arithmetic

### Definition

```
a ≡ b (mod n) if n | (a - b)

"a is congruent to b modulo n"

Equivalent: a and b have same remainder when divided by n
```

### Properties

```
1. (a + b) mod n = ((a mod n) + (b mod n)) mod n
2. (a - b) mod n = ((a mod n) - (b mod n)) mod n
3. (a × b) mod n = ((a mod n) × (b mod n)) mod n
4. aᵏ mod n = ((a mod n)ᵏ) mod n
```

### Examples

```
17 ≡ 5 (mod 12) because 17 - 5 = 12

Clock arithmetic:
23:00 ≡ 11:00 (mod 12)

Last digit: n mod 10
```

---

## 6. Modular Inverse

### Definition

```
a⁻¹ mod n is the number x such that:
a × x ≡ 1 (mod n)

Exists iff gcd(a, n) = 1
```

### Finding Inverse

```
Use Extended Euclidean Algorithm

Example: Find 3⁻¹ mod 11
3 × 4 = 12 ≡ 1 (mod 11)
So 3⁻¹ ≡ 4 (mod 11)
```

---

## 7. Applications to Hashing

### Hash Functions

```
Map large input to fixed-size output

Properties:
- Deterministic
- Fast to compute
- Uniform distribution
```

### Modular Hashing

```
h(k) = k mod m

Simple and fast
Choose m prime for better distribution
```

### Examples

```
Store keys 123, 456, 789 in table of size 10:
h(123) = 123 mod 10 = 3
h(456) = 456 mod 10 = 6
h(789) = 789 mod 10 = 9
```

---

## 💻 Python Code Examples

```python
# === Divisibility Check ===

def divides(a, b):
    """Check if a divides b"""
    return b % a == 0

print("Divisibility:")
print(f"3 | 12: {divides(3, 12)}")
print(f"5 | 12: {divides(5, 12)}")

# === Prime Testing ===

def is_prime(n):
    """Check if n is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

print("\nPrime Testing:")
for n in [2, 3, 4, 17, 20, 23, 100]:
    print(f"{n}: {is_prime(n)}")

# === Prime Factorization ===

def prime_factorization(n):
    """Return prime factorization of n"""
    factors = {}
    
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors

print("\nPrime Factorization:")
for n in [12, 60, 100, 97]:
    factors = prime_factorization(n)
    factorization = " × ".join(f"{p}^{e}" if e > 1 else str(p) 
                               for p, e in sorted(factors.items()))
    print(f"{n} = {factorization}")

# === GCD and LCM ===

def gcd(a, b):
    """Euclidean algorithm for GCD"""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """LCM using GCD"""
    return abs(a * b) // gcd(a, b)

print("\nGCD and LCM:")
print(f"gcd(48, 18) = {gcd(48, 18)}")
print(f"gcd(17, 19) = {gcd(17, 19)}")
print(f"lcm(4, 6) = {lcm(4, 6)}")
print(f"lcm(3, 5) = {lcm(3, 5)}")

# === Extended Euclidean Algorithm ===

def extended_gcd(a, b):
    """Extended Euclidean Algorithm
    Returns (gcd, x, y) such that ax + by = gcd
    """
    if a == 0:
        return b, 0, 1
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd, x, y

print("\nExtended Euclidean Algorithm:")
g, x, y = extended_gcd(48, 18)
print(f"gcd(48, 18) = {g}")
print(f"48({x}) + 18({y}) = {48*x + 18*y}")

# === Modular Arithmetic ===

def mod_inverse(a, m):
    """Find modular inverse of a mod m"""
    g, x, _ = extended_gcd(a % m, m)
    
    if g != 1:
        return None  # Inverse doesn't exist
    
    return (x % m + m) % m

print("\nModular Arithmetic:")
print(f"17 mod 12 = {17 % 12}")
print(f"3⁻¹ mod 11 = {mod_inverse(3, 11)}")
print(f"5⁻¹ mod 7 = {mod_inverse(5, 7)}")

# Verify
a, m = 3, 11
inv = mod_inverse(a, m)
print(f"Verification: {a} × {inv} mod {m} = {(a * inv) % m}")

# === Modular Exponentiation ===

def mod_exp(base, exp, mod):
    """Fast modular exponentiation"""
    result = 1
    base = base % mod
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    
    return result

print("\nModular Exponentiation:")
print(f"3¹⁷ mod 7 = {mod_exp(3, 17, 7)}")
print(f"2¹⁰⁰ mod 10 = {mod_exp(2, 100, 10)}")

# === Hash Function ===

def simple_hash(key, table_size):
    """Simple modular hash function"""
    return key % table_size

print("\nHash Function (table size = 10):")
keys = [123, 456, 789, 100, 55]
for key in keys:
    print(f"h({key}) = {simple_hash(key, 10)}")

# === Prime Generator ===

def sieve_of_eratosthenes(n):
    """Generate all primes up to n"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]

print("\nPrimes up to 50:")
print(sieve_of_eratosthenes(50))
```

---

## 📊 Summary Tables

### Divisibility Rules

| Divisor | Rule |
|---------|------|
| 2 | Last digit even |
| 3 | Sum of digits divisible by 3 |
| 5 | Last digit 0 or 5 |
| 10 | Last digit 0 |

### Number Theory Formulas

| Concept | Formula |
|---------|---------|
| GCD-LCM | gcd(a,b) × lcm(a,b) = \|ab\| |
| Modular | (a + b) mod n = ((a mod n) + (b mod n)) mod n |
| Inverse | a × a⁻¹ ≡ 1 (mod n) |

---

## 🎯 ML Applications

| Application | Number Theory Concept |
|-------------|----------------------|
| **Hashing** | Modular arithmetic |
| **Cryptography** | Prime factorization, modular inverse |
| **Random Number Generation** | Modular arithmetic |
| **Data Structures** | Hash tables (mod hashing) |

---

**Status:** ✅ Complete
**Next:** Combinatorics
