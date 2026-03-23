# 🎯 Complete AI/ML Mastery Roadmap: Zero to Hero ✅

## 📋 Overview & Learning Path Structure

```
Phase 1: Foundations (Months 1-4)
├── Mathematics Foundation
├── Programming Fundamentals (Python)
└── Python for Data Science

Phase 2: Data & Algorithms (Months 5-7)
├── Data Engineering Basics
├── DSA for AI/ML Engineers
└── Machine Learning Fundamentals

Phase 3: Core ML & Deep Learning (Months 8-12)
├── Classical ML Algorithms
├── Deep Learning Fundamentals
└── Deep Learning Frameworks

Phase 4: Specialization (Months 13-18)
├── Natural Language Processing
└── Computer Vision

Phase 5: Production & Advanced (Months 19-24)
├── MLOps & Production
├── Advanced Topics
└── Multi-Agent Systems & Modern Architectures

Phase 6: Professional Development (Ongoing)
├── Tools & Technologies
└── Projects & Portfolio
```

---

# PHASE 1: FOUNDATIONS (Months 1-4)

## 1. Mathematics Foundation

### 1.1 Linear Algebra (4-6 weeks)

**Prerequisites:** Basic high school algebra

**Core Topics:**

#### 1.1.1 Vectors and Vector Spaces
- [ ] Vector definition and representation (geometric vs algebraic)
- [ ] Vector operations: addition, subtraction, scalar multiplication
- [ ] Dot product (inner product) and geometric interpretation
- [ ] Cross product (in 3D) and applications
- [ ] Vector magnitude (norm) and unit vectors
- [ ] Linear combinations of vectors
- [ ] Linear independence and dependence
- [ ] Span of a set of vectors
- [ ] Basis vectors and coordinate systems
- [ ] Vector spaces and subspaces
- [ ] Null space, column space, row space

#### 1.1.2 Matrices and Matrix Operations
- [ ] Matrix representation and notation
- [ ] Matrix types: square, diagonal, identity, symmetric, skew-symmetric
- [ ] Matrix operations: addition, subtraction, multiplication
- [ ] Scalar multiplication of matrices
- [ ] Transpose of a matrix and properties
- [ ] Trace of a matrix
- [ ] Matrix multiplication properties (associative, distributive, NOT commutative)
- [ ] Block matrices and block operations
- [ ] Elementary matrix operations
- [ ] Permutation matrices
- [ ] Outer product of vectors

#### 1.1.3 Systems of Linear Equations
- [ ] Representation of linear systems as Ax = b
- [ ] Gaussian elimination
- [ ] Gauss-Jordan elimination
- [ ] Row echelon form (REF)
- [ ] Reduced row echelon form (RREF)
- [ ] Pivot positions and pivot columns
- [ ] Free variables and basic variables
- [ ] Existence and uniqueness of solutions
- [ ] Homogeneous systems (Ax = 0)
- [ ] Particular and general solutions

#### 1.1.4 Matrix Inverse and Determinants
- [ ] Identity matrix properties
- [ ] Invertible matrices and inverse definition
- [ ] Computing inverse using Gaussian elimination
- [ ] Computing inverse using adjugate matrix
- [ ] Properties of matrix inverse
- [ ] Determinant definition and computation (2x2, 3x3, nxn)
- [ ] Cofactor expansion (Laplace expansion)
- [ ] Properties of determinants
- [ ] Relationship between determinant and invertibility
- [ ] Cramer's rule
- [ ] Matrix determinant lemma

#### 1.1.5 Eigenvalues and Eigenvectors
- [ ] Definition and geometric interpretation
- [ ] Characteristic equation: det(A - λI) = 0
- [ ] Computing eigenvalues and eigenvectors
- [ ] Eigenspaces
- [ ] Algebraic and geometric multiplicity
- [ ] Diagonalization of matrices
- [ ] Conditions for diagonalizability
- [ ] Diagonalization process: A = PDP^(-1)
- [ ] Powers of diagonalizable matrices
- [ ] Applications to differential equations
- [ ] Spectral theorem for symmetric matrices

#### 1.1.6 Orthogonality and Projections
- [ ] Orthogonal and orthonormal vectors
- [ ] Orthogonal complement
- [ ] Orthogonal projection onto a vector
- [ ] Orthogonal projection onto a subspace
- [ ] Projection matrices and properties
- [ ] Gram-Schmidt orthogonalization process
- [ ] QR decomposition
- [ ] Least squares approximation
- [ ] Normal equations: A^T A x = A^T b
- [ ] Applications to data fitting

#### 1.1.7 Singular Value Decomposition (SVD)
- [ ] Singular values and singular vectors
- [ ] Full SVD vs reduced SVD
- [ ] Computing SVD
- [ ] Relationship to eigenvalues
- [ ] Geometric interpretation of SVD
- [ ] Low-rank approximations
- [ ] Eckart-Young theorem
- [ ] Applications: image compression, noise reduction
- [ ] Principal Component Analysis connection

#### 1.1.8 Advanced Topics
- [ ] Positive definite and positive semidefinite matrices
- [ ] Cholesky decomposition
- [ ] LU decomposition
- [ ] LDU decomposition
- [ ] Matrix norms (Frobenius, spectral, etc.)
- [ ] Condition number and numerical stability
- [ ] Pseudoinverse (Moore-Penrose)
- [ ] Jordan normal form (conceptual)
- [ ] Cayley-Hamilton theorem

**Applications in ML:**
- [ ] PCA uses eigendecomposition
- [ ] SVD for recommendation systems
- [ ] Matrix factorization techniques
- [ ] Neural network weight initialization
- [ ] Covariance matrices in statistics

---

### 1.2 Calculus (4-5 weeks)

**Prerequisites:** Basic algebra, functions

**Core Topics:**

#### 1.2.1 Functions and Limits
- [ ] Function types: polynomial, exponential, logarithmic, trigonometric
- [ ] Composite functions and inverse functions
- [ ] Limit definition and intuition
- [ ] One-sided limits
- [ ] Infinite limits and limits at infinity
- [ ] Continuity and types of discontinuity
- [ ] Intermediate Value Theorem
- [ ] Squeeze theorem
- [ ] Asymptotes (vertical, horizontal, oblique)

#### 1.2.2 Differentiation Fundamentals
- [ ] Derivative as a limit: f'(x) = lim(h→0) [f(x+h) - f(x)]/h
- [ ] Derivative as instantaneous rate of change
- [ ] Derivative as slope of tangent line
- [ ] Notation: f'(x), dy/dx, D_x f
- [ ] Differentiation rules: Power rule, Constant multiple rule, Sum/difference rule, Product rule, Quotient rule, Chain rule
- [ ] Derivatives of elementary functions
- [ ] Implicit differentiation
- [ ] Logarithmic differentiation
- [ ] Higher-order derivatives

#### 1.2.3 Applications of Derivatives
- [ ] Critical points and stationary points
- [ ] First derivative test for extrema
- [ ] Second derivative test
- [ ] Concavity and inflection points
- [ ] Curve sketching
- [ ] Optimization problems
- [ ] Related rates
- [ ] Mean Value Theorem
- [ ] Rolle's Theorem
- [ ] L'Hôpital's rule for indeterminate forms

#### 1.2.4 Multivariable Calculus - Partial Derivatives
- [ ] Functions of several variables
- [ ] Domain and range in multiple dimensions
- [ ] Limits and continuity in multiple variables
- [ ] Partial derivatives: ∂f/∂x, ∂f/∂y
- [ ] Higher-order partial derivatives
- [ ] Clairaut's theorem (equality of mixed partials)
- [ ] Tangent planes and linear approximation
- [ ] Differentiability in multiple variables

#### 1.2.5 Gradient and Directional Derivatives
- [ ] Gradient vector: ∇f = (∂f/∂x, ∂f/∂y, ...)
- [ ] Geometric interpretation of gradient
- [ ] Directional derivatives
- [ ] Maximum rate of increase direction
- [ ] Level curves and level surfaces
- [ ] Gradient perpendicular to level curves
- [ ] Applications to optimization

#### 1.2.6 Chain Rule for Multivariable Functions
- [ ] Chain rule for composite functions
- [ ] Tree diagrams for chain rule
- [ ] Implicit differentiation in multiple variables
- [ ] Total differential
- [ ] Error propagation

#### 1.2.7 Optimization in Multiple Variables
- [ ] Critical points in multiple variables
- [ ] Second partial derivative test
- [ ] Hessian matrix
- [ ] Classification of critical points (min, max, saddle)
- [ ] Constrained optimization
- [ ] Lagrange multipliers (SINGLE and MULTIPLE constraints)
- [ ] KKT conditions (introduction)

#### 1.2.8 Integration Fundamentals
- [ ] Antiderivatives and indefinite integrals
- [ ] Definite integrals and area under curve
- [ ] Fundamental Theorem of Calculus (Parts 1 & 2)
- [ ] Integration techniques: Substitution, Integration by parts, Trigonometric substitution, Partial fraction decomposition
- [ ] Improper integrals
- [ ] Numerical integration (Riemann sums, trapezoidal rule, Simpson's rule)

#### 1.2.9 Multiple Integration
- [ ] Double integrals over rectangles
- [ ] Double integrals over general regions
- [ ] Iterated integrals
- [ ] Change of order of integration
- [ ] Double integrals in polar coordinates
- [ ] Triple integrals
- [ ] Change of variables in multiple integrals
- [ ] Jacobian determinant

#### 1.2.10 Vector Calculus (Essentials for ML)
- [ ] Vector fields
- [ ] Gradient, divergence, curl
- [ ] Line integrals
- [ ] Surface integrals
- [ ] Green's theorem (conceptual)
- [ ] Stokes' theorem (conceptual)
- [ ] Divergence theorem (conceptual)

#### 1.2.11 Advanced Topics for ML
- [ ] Taylor series and Taylor polynomials
- [ ] Multivariate Taylor expansion (CRITICAL for optimization)
- [ ] Newton's method for optimization
- [ ] Convex functions and convex optimization basics
- [ ] Jensen's inequality
- [ ] Lipschitz continuity
- [ ] Uniform continuity

**Applications in ML:**
- [ ] Gradient descent uses gradients
- [ ] Backpropagation uses chain rule extensively
- [ ] Loss function optimization
- [ ] Taylor expansion for approximation
- [ ] Hessian in second-order optimization methods

---

### 1.3 Probability & Statistics (5-6 weeks)

**Prerequisites:** Basic algebra, set theory

**Core Topics:**

#### 1.3.1 Foundations of Probability
- [ ] Random experiments and sample spaces
- [ ] Events and event operations (union, intersection, complement)
- [ ] Axioms of probability (Kolmogorov axioms)
- [ ] Classical, empirical, and subjective probability
- [ ] Counting principles: Multiplication rule, Permutations, Combinations, Binomial coefficients
- [ ] Conditional probability: P(A|B)
- [ ] Multiplication rule: P(A∩B) = P(A|B)P(B)
- [ ] Law of Total Probability
- [ ] Bayes' Theorem (CRITICAL for Bayesian methods)
- [ ] Independent events

#### 1.3.2 Random Variables
- [ ] Definition of random variables
- [ ] Discrete vs continuous random variables
- [ ] Probability mass function (PMF) for discrete RVs
- [ ] Probability density function (PDF) for continuous RVs
- [ ] Cumulative distribution function (CDF)
- [ ] Properties of CDF
- [ ] Survival function
- [ ] Quantile function (inverse CDF)
- [ ] Mixed distributions

#### 1.3.3 Expectation and Moments
- [ ] Expected value (mean) definition
- [ ] Expected value of functions of RVs
- [ ] Linearity of expectation
- [ ] Variance and standard deviation
- [ ] Properties of variance
- [ ] Covariance between two RVs
- [ ] Correlation coefficient
- [ ] Moments: raw moments, central moments
- [ ] Skewness (3rd moment)
- [ ] Kurtosis (4th moment)
- [ ] Moment generating functions (MGF)
- [ ] Characteristic functions

#### 1.3.4 Important Discrete Distributions
- [ ] Bernoulli distribution
- [ ] Binomial distribution
- [ ] Geometric distribution
- [ ] Negative binomial distribution
- [ ] Poisson distribution
- [ ] Hypergeometric distribution
- [ ] Multinomial distribution
- [ ] Properties, mean, variance, applications for each

#### 1.3.5 Important Continuous Distributions
- [ ] Uniform distribution
- [ ] Normal (Gaussian) distribution (Standard normal, Properties and symmetry, 68-95-99.7 rule, CLT connection)
- [ ] Exponential distribution
- [ ] Gamma distribution
- [ ] Beta distribution
- [ ] Chi-squared distribution
- [ ] Student's t-distribution
- [ ] F-distribution
- [ ] Log-normal distribution
- [ ] Properties, mean, variance, applications for each

#### 1.3.6 Joint Distributions
- [ ] Joint PMF and joint PDF
- [ ] Marginal distributions
- [ ] Conditional distributions
- [ ] Independent random variables
- [ ] Sum of random variables
- [ ] Convolution of distributions
- [ ] Bivariate normal distribution
- [ ] Conditional expectation
- [ ] Law of Iterated Expectations

#### 1.3.7 Limit Theorems
- [ ] Chebyshev's inequality
- [ ] Law of Large Numbers (weak and strong)
- [ ] Central Limit Theorem (CLT)
- [ ] Continuity correction

#### 1.3.8 Descriptive Statistics
- [ ] Population vs sample
- [ ] Parameters vs statistics
- [ ] Measures of central tendency: mean, median, mode
- [ ] Measures of dispersion: range, variance, standard deviation, IQR
- [ ] Measures of position: percentiles, quartiles
- [ ] Five-number summary
- [ ] Box plots and whisker plots
- [ ] Histograms and density plots
- [ ] Q-Q plots
- [ ] Outlier detection

#### 1.3.9 Sampling Distributions
- [ ] Sampling distribution of sample mean
- [ ] Sampling distribution of sample proportion
- [ ] Sampling distribution of sample variance
- [ ] t-distribution in sampling
- [ ] Chi-squared distribution in sampling
- [ ] F-distribution in sampling
- [ ] Standard error

#### 1.3.10 Estimation Theory
- [ ] Point estimation
- [ ] Properties of estimators: Unbiasedness, Consistency, Efficiency, Sufficiency
- [ ] Method of Moments (MOM)
- [ ] Maximum Likelihood Estimation (MLE)
- [ ] Confidence intervals (mean, proportion, variance, difference of means, sample size)
- [ ] Bayesian estimation basics (Prior, Posterior, Conjugate priors, Credible intervals)

#### 1.3.11 Hypothesis Testing
- [ ] Null and alternative hypotheses
- [ ] Type I and Type II errors
- [ ] Significance level (α)
- [ ] Power of a test (1-β)
- [ ] p-values and interpretation
- [ ] One-tailed vs two-tailed tests
- [ ] z-tests (one sample, two sample)
- [ ] t-tests: One-sample, Independent two-sample, Paired, Welch's
- [ ] Chi-squared tests: Goodness of fit, Test of independence, Test of homogeneity
- [ ] F-tests
- [ ] ANOVA: One-way, Two-way, MANOVA (conceptual)
- [ ] Post-hoc tests (Tukey, Bonferroni)
- [ ] Non-parametric tests: Sign test, Wilcoxon, Mann-Whitney U, Kruskal-Wallis
- [ ] Multiple testing correction: Bonferroni, FDR, Benjamini-Hochberg

#### 1.3.12 Regression Analysis
- [ ] Simple linear regression
- [ ] Least squares estimation
- [ ] Assumptions of linear regression
- [ ] Coefficient of determination (R²)
- [ ] Correlation vs causation
- [ ] Residual analysis
- [ ] Confidence intervals for regression coefficients
- [ ] Prediction intervals
- [ ] Multiple linear regression
- [ ] Matrix formulation of regression
- [ ] Polynomial regression

#### 1.3.13 Bayesian Statistics
- [ ] Bayes' theorem review
- [ ] Prior selection (informative, non-informative, Jeffreys)
- [ ] Likelihood function
- [ ] Posterior computation
- [ ] Conjugate priors: Beta-Binomial, Gamma-Poisson, Normal-Normal
- [ ] Bayesian inference
- [ ] Bayesian hypothesis testing
- [ ] Bayes factors
- [ ] MCMC concepts
- [ ] Gibbs sampling (introduction)
- [ ] Metropolis-Hastings algorithm (introduction)

#### 1.3.14 Advanced Topics for ML
- [ ] Information Theory: Entropy, Cross-entropy, KL divergence, Mutual information
- [ ] Copulas
- [ ] Order statistics
- [ ] Extreme value theory
- [ ] Time series basics (stationarity, autocorrelation)
- [ ] Multivariate distributions
- [ ] Wishart distribution
- [ ] Dirichlet distribution (for topic models)

**Applications in ML:**
- [ ] MLE for model training
- [ ] Bayesian methods for uncertainty
- [ ] Hypothesis testing for A/B testing
- [ ] Distributions for generative models
- [ ] Information theory for decision trees

---

### 1.4 Discrete Mathematics (2-3 weeks, optional but recommended)

**Prerequisites:** Basic algebra

**Core Topics:**

#### 1.4.1 Logic and Proofs
- [ ] Propositional logic
- [ ] Truth tables
- [ ] Logical equivalences
- [ ] Predicates and quantifiers
- [ ] Nested quantifiers
- [ ] Rules of inference
- [ ] Proof methods: Direct proof, Contraposition, Contradiction, Cases, Mathematical induction, Strong induction, Structural induction

#### 1.4.2 Set Theory
- [ ] Sets and set notation
- [ ] Set operations: union, intersection, difference, complement
- [ ] Venn diagrams
- [ ] Power sets
- [ ] Cartesian products
- [ ] Set identities
- [ ] Partitions
- [ ] Russell's paradox (conceptual)

#### 1.4.3 Functions and Relations
- [ ] Functions: domain, codomain, range
- [ ] Injective, surjective, bijective functions
- [ ] Inverse functions
- [ ] Composition of functions
- [ ] Floor and ceiling functions
- [ ] Relations and properties
- [ ] Equivalence relations
- [ ] Equivalence classes
- [ ] Partial orderings
- [ ] Hasse diagrams
- [ ] Lattices

#### 1.4.4 Number Theory Basics
- [ ] Divisibility
- [ ] Prime numbers
- [ ] Fundamental theorem of arithmetic
- [ ] GCD and LCM
- [ ] Euclidean algorithm
- [ ] Modular arithmetic
- [ ] Congruences
- [ ] Applications to hashing

#### 1.4.5 Combinatorics
- [ ] Sum and product rules
- [ ] Permutations and combinations
- [ ] Binomial theorem
- [ ] Pascal's triangle
- [ ] Inclusion-exclusion principle
- [ ] Pigeonhole principle
- [ ] Generating functions (introduction)
- [ ] Recurrence relations

#### 1.4.6 Graph Theory
- [ ] Graph definitions: vertices, edges
- [ ] Types of graphs: directed, undirected, weighted
- [ ] Degree of vertices
- [ ] Paths and cycles
- [ ] Connected components
- [ ] Eulerian and Hamiltonian paths
- [ ] Graph representations: adjacency matrix, adjacency list
- [ ] Trees and properties
- [ ] Spanning trees
- [ ] Minimum spanning trees (Kruskal's, Prim's)
- [ ] Graph coloring
- [ ] Planar graphs
- [ ] Bipartite graphs
- [ ] Matching

#### 1.4.7 Boolean Algebra
- [ ] Boolean operations
- [ ] Boolean expressions
- [ ] Logic gates
- [ ] Karnaugh maps
- [ ] Digital circuit basics

**Applications in ML:**
- [ ] Graph theory for GNNs
- [ ] Combinatorics for probability
- [ ] Logic for rule-based systems
- [ ] Set theory for data structures

---

## 2. Programming Fundamentals (Python) (6-8 weeks)

**Prerequisites:** None (absolute beginner friendly)

### 2.1 Python Basics (2 weeks)

#### 2.1.1 Introduction to Programming
- [ ] What is programming?
- [ ] Compiled vs interpreted languages
- [ ] Python history and versions (Python 3.x)
- [ ] Setting up Python environment
- [ ] Running Python scripts
- [ ] Interactive Python shell (REPL)
- [ ] Comments and documentation

#### 2.1.2 Variables and Data Types
- [ ] Variables and assignment
- [ ] Dynamic typing
- [ ] Basic data types: int, float, str, bool, None
- [ ] Type conversion (casting)
- [ ] Type checking with type()
- [ ] f-strings and string formatting
- [ ] Escape sequences

#### 2.1.3 Operators
- [ ] Arithmetic operators: +, -, *, /, //, %, **
- [ ] Comparison operators: ==, !=, <, >, <=, >=
- [ ] Logical operators: and, or, not
- [ ] Bitwise operators: &, |, ^, ~, <<, >>
- [ ] Assignment operators: =, +=, -=, *=, /=, etc.
- [ ] Identity operators: is, is not
- [ ] Membership operators: in, not in
- [ ] Operator precedence

#### 2.1.4 Control Structures - Conditionals
- [ ] if statements
- [ ] elif clauses
- [ ] else clauses
- [ ] Nested conditionals
- [ ] Ternary operator (conditional expression)
- [ ] Truthy and falsy values
- [ ] Short-circuit evaluation

#### 2.1.5 Control Structures - Loops
- [ ] while loops
- [ ] for loops
- [ ] range() function
- [ ] Loop control: break, continue, pass
- [ ] else clause in loops
- [ ] Nested loops
- [ ] Iterating over sequences
- [ ] enumerate() function
- [ ] zip() function

---

### 2.2 Data Structures (2 weeks)

#### 2.2.1 Strings
- [ ] String creation and indexing
- [ ] String slicing
- [ ] String immutability
- [ ] String methods: Case, Search, Replace, Split/join, Strip, Check, Format, Encode/decode
- [ ] String formatting: %, .format(), f-strings
- [ ] Raw strings and Unicode
- [ ] Regular expressions basics (re module)

#### 2.2.2 Lists
- [ ] List creation and initialization
- [ ] List indexing and slicing
- [ ] List mutability
- [ ] List methods: Add, Remove, Query, Sort, Reverse, Copy
- [ ] List comprehensions
- [ ] Nested lists
- [ ] Shallow vs deep copy
- [ ] List as stack and queue

#### 2.2.3 Tuples
- [ ] Tuple creation
- [ ] Tuple immutability
- [ ] Tuple packing and unpacking
- [ ] Named tuples (collections.namedtuple)
- [ ] When to use tuples vs lists
- [ ] Tuple methods: count(), index()

#### 2.2.4 Sets
- [ ] Set creation
- [ ] Set mutability
- [ ] Set operations: Union, Intersection, Difference, Symmetric difference, Subset/superset
- [ ] Set methods: add(), remove(), discard(), pop(), clear()
- [ ] Set comprehensions
- [ ] Frozen sets (immutable sets)
- [ ] Set applications

#### 2.2.5 Dictionaries
- [ ] Dictionary creation
- [ ] Key-value pairs
- [ ] Dictionary access and modification
- [ ] Dictionary methods: Access, Update, Remove, Copy
- [ ] Dictionary comprehensions
- [ ] Nested dictionaries
- [ ] OrderedDict, defaultdict, Counter
- [ ] Dictionary view objects
- [ ] Hashable vs unhashable types

---

### 2.3 Functions and Modules (1.5 weeks)

#### 2.3.1 Functions
- [ ] Defining functions with def
- [ ] Function arguments and parameters
- [ ] Return statement
- [ ] Function scope: local, enclosing, global, built-in (LEGB)
- [ ] global and nonlocal keywords
- [ ] Default parameter values
- [ ] Keyword arguments
- [ ] Variable-length arguments: *args, **kwargs
- [ ] Positional-only and keyword-only arguments
- [ ] Function annotations
- [ ] Lambda functions (anonymous functions)
- [ ] Higher-order functions
- [ ] Function decorators basics
- [ ] Recursion basics
- [ ] Docstrings and documentation

#### 2.3.2 Modules and Packages
- [ ] What are modules?
- [ ] Importing modules: import, from ... import
- [ ] Module search path (sys.path)
- [ ] Creating custom modules
- [ ] The if __name__ == "__main__" idiom
- [ ] Packages and __init__.py
- [ ] Subpackages
- [ ] Relative vs absolute imports
- [ ] Standard library overview: os, sys, math, random, statistics, datetime, time, collections, itertools, functools, pathlib, json, csv, re, typing
- [ ] Installing packages with pip
- [ ] requirements.txt
- [ ] Virtual environments (venv, virtualenv)
- [ ] conda environments
- [ ] pipenv and poetry

---

### 2.4 Object-Oriented Programming (1.5 weeks)

#### 2.4.1 OOP Fundamentals
- [ ] Classes and objects
- [ ] Class definition with class keyword
- [ ] The __init__() constructor
- [ ] self parameter
- [ ] Instance attributes vs class attributes
- [ ] Instance methods
- [ ] The __str__() and __repr__() methods
- [ ] Deleting objects: del, __del__()

#### 2.4.2 Encapsulation
- [ ] Public, protected, private attributes
- [ ] Name mangling
- [ ] Getters and setters
- [ ] @property decorator
- [ ] @<attribute>.setter decorator
- [ ] @<attribute>.deleter decorator
- [ ] Data hiding principles

#### 2.4.3 Inheritance
- [ ] Base classes and derived classes
- [ ] Creating child classes
- [ ] The super() function
- [ ] Overriding methods
- [ ] Multiple inheritance
- [ ] Method Resolution Order (MRO)
- [ ] isinstance() and issubclass()
- [ ] Abstract base classes (abc module)
- [ ] @abstractmethod decorator

#### 2.4.4 Polymorphism
- [ ] Polymorphism concept
- [ ] Duck typing
- [ ] Operator overloading
- [ ] Magic/dunder methods: Arithmetic, Comparison, String, Container, Context manager, Callable
- [ ] Design patterns basics: Singleton, Factory, Observer, Strategy

#### 2.4.5 Advanced OOP
- [ ] Class methods (@classmethod)
- [ ] Static methods (@staticmethod)
- [ ] Class decorators
- [ ] Metaclasses (introduction)
- [ ] Mixins
- [ ] Composition vs inheritance

---

### 2.5 Functional Programming Concepts (1 week)

#### 2.5.1 Functional Programming Basics
- [ ] Pure functions
- [ ] Immutability
- [ ] First-class and higher-order functions
- [ ] Function composition
- [ ] Referential transparency

#### 2.5.2 Built-in Functional Tools
- [ ] map() function
- [ ] filter() function
- [ ] reduce() function (functools.reduce)
- [ ] all() and any()
- [ ] sum(), min(), max()

#### 2.5.3 itertools Module
- [ ] Infinite iterators: count(), cycle(), repeat()
- [ ] Finite iterators: accumulate(), chain(), compress(), dropwhile(), takewhile()
- [ ] Combinatoric iterators: product(), permutations(), combinations(), combinations_with_replacement()

#### 2.5.4 functools Module
- [ ] partial functions (functools.partial)
- [ ] lru_cache decorator
- [ ] wraps decorator
- [ ] singledispatch
- [ ] reduce function

#### 2.5.5 Generators and Iterators
- [ ] Iterables vs iterators
- [ ] The iter() and next() functions
- [ ] Creating iterators with classes
- [ ] Generator functions (yield)
- [ ] Generator expressions
- [ ] yield from
- [ ] Coroutines basics
- [ ] send(), throw(), close() methods

---

### 2.6 File Handling and Error Handling (1 week)

#### 2.6.1 File Operations
- [ ] Opening files: open() function
- [ ] File modes: r, w, a, r+, w+, a+, b, t
- [ ] Reading files: read(), readline(), readlines()
- [ ] Writing files: write(), writelines()
- [ ] Closing files
- [ ] Context managers (with statement)
- [ ] File position: seek(), tell()
- [ ] Working with binary files
- [ ] Working with CSV files (csv module)
- [ ] Working with JSON files (json module)
- [ ] Working with pickle files (pickle module)
- [ ] pathlib for modern path handling

#### 2.6.2 Exception Handling
- [ ] Exceptions vs syntax errors
- [ ] try-except blocks
- [ ] Handling multiple exceptions
- [ ] else clause in try-except
- [ ] finally clause
- [ ] Raising exceptions: raise
- [ ] Exception chaining
- [ ] Custom exception classes
- [ ] Built-in exception hierarchy
- [ ] Assertions
- [ ] Logging basics (logging module)
- [ ] Debugging with pdb

---

### 2.7 Advanced Python Topics (1 week)

#### 2.7.1 Decorators
- [ ] Function decorators
- [ ] Decorator syntax (@decorator)
- [ ] Decorators with arguments
- [ ] Multiple decorators
- [ ] Class decorators
- [ ] Built-in decorators: @staticmethod, @classmethod, @property, @functools.wraps

#### 2.7.2 Context Managers
- [ ] with statement
- [ ] Creating context managers with classes
- [ ] Creating context managers with contextlib
- [ ] @contextmanager decorator
- [ ] ExitStack

#### 2.7.3 Type Hints
- [ ] Type annotations
- [ ] typing module
- [ ] Generic types
- [ ] Type aliases
- [ ] Union, Optional, List, Dict, Tuple, Set
- [ ] Callable types
- [ ] Type checking with mypy

#### 2.7.4 Concurrency Basics
- [ ] GIL (Global Interpreter Lock)
- [ ] Threading basics (threading module)
- [ ] Multiprocessing basics (multiprocessing module)
- [ ] asyncio basics
- [ ] Concurrent.futures

#### 2.7.5 Testing
- [ ] Unit testing principles
- [ ] unittest module
- [ ] pytest basics
- [ ] Test fixtures
- [ ] Mocking (unittest.mock)
- [ ] Test-driven development concepts

---

## 3. Python for Data Science (4-5 weeks)

**Prerequisites:** Python fundamentals, basic mathematics

### 3.1 Jupyter Notebooks (3-4 days)

#### 3.1.1 Jupyter Basics
- [ ] What is Jupyter?
- [ ] Jupyter Notebook vs JupyterLab
- [ ] Installation and setup
- [ ] Creating and running notebooks
- [ ] Cell types: code, markdown, raw
- [ ] Keyboard shortcuts
- [ ] Magic commands: %timeit, %matplotlib, %load, etc.
- [ ] Line magic vs cell magic

#### 3.1.2 Notebook Features
- [ ] Markdown formatting in notebooks
- [ ] Embedding images, videos, LaTeX
- [ ] Creating tables
- [ ] Code cells and output
- [ ] Cell execution order
- [ ] Restarting kernel
- [ ] Interrupting execution
- [ ] Checkpoints
- [ ] Exporting notebooks (HTML, PDF, Python)
- [ ] nbconvert

#### 3.1.3 Best Practices
- [ ] Organizing notebooks
- [ ] Documentation with markdown
- [ ] Version control for notebooks
- [ ] nbformat
- [ ] Jupyter extensions
- [ ] Widgets basics (ipywidgets)

---

### 3.2 NumPy (1.5-2 weeks)

**Prerequisites:** Python basics, linear algebra fundamentals

#### 3.2.1 NumPy Arrays
- [ ] Introduction to NumPy
- [ ] Creating arrays: np.array(), np.zeros(), np.ones(), np.full(), np.empty()
- [ ] np.arange(), np.linspace(), np.logspace()
- [ ] np.random: rand(), randn(), randint(), choice(), shuffle(), permutation()
- [ ] Array attributes: ndim, shape, size, dtype, itemsize, nbytes
- [ ] Array indexing and slicing
- [ ] Boolean indexing
- [ ] Fancy indexing
- [ ] np.where()
- [ ] Array views vs copies

#### 3.2.2 Array Operations
- [ ] Element-wise operations
- [ ] Broadcasting rules
- [ ] Universal functions (ufuncs)
- [ ] Mathematical functions: np.sin, np.cos, np.exp, np.log, np.sqrt, etc.
- [ ] Aggregation functions: np.sum, np.mean, np.std, np.var, np.min, np.max
- [ ] Axis parameter
- [ ] np.argmin, np.argmax
- [ ] np.sort, np.argsort
- [ ] np.unique
- [ ] Set operations: np.intersect1d, np.union1d, np.setdiff1d, np.setxor1d

#### 3.2.3 Array Manipulation
- [ ] Reshaping: reshape(), ravel(), flatten()
- [ ] Transposing: T, transpose(), swapaxes()
- [ ] Stacking: np.stack, np.vstack, np.hstack, np.dstack, np.column_stack
- [ ] Splitting: np.split, np.vsplit, np.hsplit, np.dsplit
- [ ] Concatenation: np.concatenate
- [ ] Adding/removing elements: np.append, np.insert, np.delete
- [ ] Tiling: np.tile, np.repeat
- [ ] Padding: np.pad

#### 3.2.4 Linear Algebra with NumPy
- [ ] Matrix multiplication: @, np.dot, np.matmul
- [ ] np.linalg module: norm, det, trace, inv, pinv, eig, eigh, svd, qr, lu, solve, lstsq, matrix_rank, cond
- [ ] Vectorized operations
- [ ] Einstein summation: np.einsum

#### 3.2.5 Advanced NumPy
- [ ] Structured arrays
- [ ] Record arrays
- [ ] Memory-mapped files
- [ ] np.vectorize
- [ ] np.apply_along_axis, np.apply_over_axes
- [ ] Strides and as_strided
- [ ] Broadcasting advanced patterns
- [ ] Performance optimization with NumPy

---

### 3.3 Pandas (2 weeks)

**Prerequisites:** NumPy, Python data structures

#### 3.3.1 Series
- [ ] Creating Series
- [ ] Series attributes: index, values, dtype
- [ ] Indexing and slicing Series
- [ ] Boolean indexing
- [ ] Series operations
- [ ] Handling missing data: isna(), notna(), dropna(), fillna()
- [ ] Series methods: head(), tail(), describe(), value_counts()

#### 3.3.2 DataFrame Basics
- [ ] Creating DataFrames
- [ ] DataFrame from dictionaries, lists, arrays
- [ ] Reading data: read_csv, read_excel, read_json, read_sql, read_html, read_parquet
- [ ] DataFrame attributes: shape, columns, index, dtypes
- [ ] Selecting columns
- [ ] Selecting rows: loc[], iloc[]
- [ ] Boolean indexing
- [ ] Setting values
- [ ] Adding/removing columns
- [ ] Dropping rows/columns: drop()

#### 3.3.3 Data Inspection
- [ ] head(), tail(), sample()
- [ ] info(), describe()
- [ ] shape, size, ndim
- [ ] columns, index
- [ ] dtypes, astype()
- [ ] value_counts()
- [ ] unique(), nunique()
- [ ] isna(), notna(), isnull(), notnull()

#### 3.3.4 Data Cleaning
- [ ] Handling missing data: Detection, Removal, Imputation
- [ ] Handling duplicates: duplicated(), drop_duplicates()
- [ ] Data type conversion: astype(), pd.to_numeric(), pd.to_datetime()
- [ ] String operations: .str accessor
- [ ] Renaming: rename(), rename_axis()
- [ ] Replacing values: replace(), map(), applymap()

#### 3.3.5 Data Transformation
- [ ] Sorting: sort_values(), sort_index()
- [ ] Ranking: rank()
- [ ] Applying functions: apply(), applymap(), map()
- [ ] Lambda functions with apply
- [ ] Vectorized operations
- [ ] Binning: cut(), qcut()
- [ ] One-hot encoding: get_dummies()
- [ ] Discretization

#### 3.3.6 Data Aggregation
- [ ] GroupBy operations: groupby()
- [ ] Split-Apply-Combine pattern
- [ ] Aggregation functions: sum, mean, count, min, max, std, var
- [ ] Multiple aggregations: agg()
- [ ] Transform operations: transform()
- [ ] Filter operations: filter()
- [ ] GroupBy with multiple columns
- [ ] GroupBy with custom functions
- [ ] pivot_table()
- [ ] crosstab()

#### 3.3.7 Merging and Joining
- [ ] concat(): concatenating DataFrames
- [ ] merge(): database-style joins (Inner, outer, left, right)
- [ ] join(): index-based joining
- [ ] Combining: combine_first()

#### 3.3.8 Time Series
- [ ] Creating datetime: pd.to_datetime(), pd.date_range()
- [ ] DatetimeIndex
- [ ] Time-based indexing and slicing
- [ ] Resampling: resample()
- [ ] Rolling windows: rolling()
- [ ] Expanding windows: expanding()
- [ ] Time zone handling
- [ ] Periods and PeriodIndex
- [ ] Timedelta

#### 3.3.9 Advanced Pandas
- [ ] MultiIndex (hierarchical indexing)
- [ ] Categorical data
- [ ] Sparse data
- [ ] Styling DataFrames
- [ ] Performance optimization: Vectorization, Using appropriate dtypes, chunking, eval() and query()
- [ ] Interoperability with NumPy
- [ ] Plotting with Pandas

---

### 3.4 Data Visualization (1.5 weeks)

#### 3.4.1 Matplotlib Fundamentals
- [ ] Matplotlib architecture: pyplot, Figure, Axes
- [ ] Creating figures: plt.figure(), plt.subplots()
- [ ] Plotting basics: plot(), scatter(), bar(), barh(), hist(), boxplot()
- [ ] Figure and axes customization
- [ ] Titles, labels, legends
- [ ] Grid, axis limits, ticks
- [ ] Saving figures: savefig()
- [ ] Multiple subplots
- [ ] Figure sizes and DPI

#### 3.4.2 Matplotlib Advanced
- [ ] Line styles, markers, colors
- [ ] Color maps (colormaps)
- [ ] Annotations: annotate(), text()
- [ ] Arrows and shapes
- [ ] Twin axes
- [ ] Inset axes
- [ ] 3D plotting (mplot3d)
- [ ] Customizing ticks and spines
- [ ] Layout engines: tight_layout, constrained_layout

#### 3.4.3 Seaborn
- [ ] Seaborn vs Matplotlib
- [ ] Setting themes: set_theme(), set_style()
- [ ] Color palettes
- [ ] Distribution plots: histplot, kdeplot, ecdfplot, rugplot, jointplot
- [ ] Categorical plots: stripplot, swarmplot, boxplot, violinplot, boxenplot, barplot, countplot, pointplot
- [ ] Relational plots: scatterplot, lineplot, relplot
- [ ] Matrix plots: heatmap, clustermap
- [ ] Regression plots: lmplot, regplot, residplot
- [ ] Pair plots: pairplot
- [ ] Facet grids: FacetGrid
- [ ] Multi-plot grids: catplot, displot, relplot

#### 3.4.4 Visualization Best Practices
- [ ] Choosing the right plot type
- [ ] Color theory for visualization
- [ ] Avoiding chart junk
- [ ] Data-ink ratio
- [ ] Accessibility considerations
- [ ] Storytelling with data
- [ ] Interactive visualization basics (plotly, bokeh)

---

# PHASE 2: DATA & ALGORITHMS (Months 5-7)

## 4. Data Engineering Basics (4-5 weeks)

**Prerequisites:** Python, basic SQL knowledge helpful

### 4.1 SQL Fundamentals (2 weeks)

#### 4.1.1 Database Basics
- [ ] What is a database?
- [ ] Relational databases concepts
- [ ] Tables, rows, columns
- [ ] Primary keys, foreign keys
- [ ] Data types in SQL
- [ ] NULL values

#### 4.1.2 Basic SQL Queries
- [ ] SELECT statement
- [ ] WHERE clause
- [ ] Comparison operators
- [ ] Logical operators: AND, OR, NOT
- [ ] IN, BETWEEN, LIKE operators
- [ ] ORDER BY
- [ ] LIMIT/OFFSET (or TOP, FETCH)
- [ ] DISTINCT
- [ ] Comments in SQL

#### 4.1.3 Aggregation and Grouping
- [ ] Aggregate functions: COUNT, SUM, AVG, MIN, MAX
- [ ] GROUP BY clause
- [ ] HAVING clause
- [ ] GROUP BY with multiple columns
- [ ] ROLLUP, CUBE, GROUPING SETS

#### 4.1.4 Joins
- [ ] INNER JOIN
- [ ] LEFT JOIN (LEFT OUTER JOIN)
- [ ] RIGHT JOIN (RIGHT OUTER JOIN)
- [ ] FULL OUTER JOIN
- [ ] CROSS JOIN
- [ ] SELF JOIN
- [ ] Joining multiple tables
- [ ] Join conditions

#### 4.1.5 Subqueries
- [ ] Subqueries in WHERE
- [ ] Subqueries in SELECT
- [ ] Subqueries in FROM
- [ ] Correlated subqueries
- [ ] EXISTS and NOT EXISTS
- [ ] IN with subqueries
- [ ] ANY, ALL operators

#### 4.1.6 Set Operations
- [ ] UNION
- [ ] UNION ALL
- [ ] INTERSECT
- [ ] EXCEPT (or MINUS)

#### 4.1.7 Data Modification
- [ ] INSERT INTO
- [ ] UPDATE
- [ ] DELETE
- [ ] TRUNCATE
- [ ] INSERT ... SELECT
- [ ] UPSERT (INSERT ... ON CONFLICT / ON DUPLICATE KEY)

#### 4.1.8 Schema and Table Management
- [ ] CREATE TABLE
- [ ] ALTER TABLE
- [ ] DROP TABLE
- [ ] CREATE INDEX
- [ ] DROP INDEX
- [ ] Constraints: PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK, NOT NULL, DEFAULT
- [ ] CREATE VIEW
- [ ] DROP VIEW
- [ ] Temporary tables
- [ ] Common Table Expressions (CTEs)
- [ ] WITH clause

#### 4.1.9 Advanced SQL
- [ ] Window functions: ROW_NUMBER(), RANK(), DENSE_RANK(), NTILE(), LEAD(), LAG(), FIRST_VALUE(), LAST_VALUE(), NTH_VALUE()
- [ ] OVER clause
- [ ] PARTITION BY
- [ ] ORDER BY in window functions
- [ ] Window frames: ROWS, RANGE
- [ ] Stored procedures basics
- [ ] Functions (UDFs)
- [ ] Triggers basics
- [ ] Transactions: BEGIN, COMMIT, ROLLBACK
- [ ] ACID properties
- [ ] Isolation levels
- [ ] Indexes and performance
- [ ] Query optimization basics
- [ ] EXPLAIN and query plans

#### 4.1.10 SQL for Data Science
- [ ] Date/time functions
- [ ] String functions
- [ ] Mathematical functions
- [ ] CASE expressions
- [ ] Pivoting data
- [ ] Unpivoting data
- [ ] Sampling in SQL
- [ ] Percentiles and quantiles
- [ ] Running totals
- [ ] Moving averages

---

### 4.2 NoSQL Basics (1 week)

#### 4.2.1 NoSQL Overview
- [ ] SQL vs NoSQL
- [ ] CAP theorem
- [ ] BASE vs ACID
- [ ] Types of NoSQL databases

#### 4.2.2 Document Databases (MongoDB)
- [ ] Document model
- [ ] Collections and documents
- [ ] BSON format
- [ ] Basic CRUD operations
- [ ] Querying documents
- [ ] Indexing
- [ ] Aggregation pipeline
- [ ] When to use document databases

#### 4.2.3 Key-Value Stores (Redis)
- [ ] Key-value model
- [ ] Redis data types: strings, lists, sets, hashes, sorted sets
- [ ] Basic operations
- [ ] Use cases: caching, sessions, real-time

#### 4.2.4 Column-Family Stores (Cassandra, HBase)
- [ ] Column-family model
- [ ] Wide-column stores
- [ ] Partition keys, clustering keys
- [ ] Read/write patterns
- [ ] Use cases: time series, big data

#### 4.2.5 Graph Databases (Neo4j)
- [ ] Graph model: nodes, edges, properties
- [ ] Cypher query language basics
- [ ] Pattern matching
- [ ] Use cases: social networks, recommendations, fraud detection

---

### 4.3 Data Preprocessing and Cleaning (1.5 weeks)

#### 4.3.1 Data Quality
- [ ] Dimensions of data quality
- [ ] Data profiling
- [ ] Identifying data issues
- [ ] Data validation rules

#### 4.3.2 Handling Missing Data
- [ ] Types of missing data: MCAR, MAR, MNAR
- [ ] Detection of missing values
- [ ] Deletion methods: listwise, pairwise
- [ ] Imputation methods: Mean/median/mode, KNN, Regression, Multiple imputation, Forward/backward fill
- [ ] Missing indicator variables

#### 4.3.3 Handling Outliers
- [ ] Detecting outliers: Z-score, IQR, Isolation Forest, DBSCAN
- [ ] Treating outliers: Capping/flooring, Transformation, Removal, Binning

#### 4.3.4 Data Transformation
- [ ] Scaling: Min-Max, Standardization, Robust scaling, MaxAbs scaling
- [ ] Normalization: L1, L2
- [ ] Transformations: Log, Square root, Box-Cox, Yeo-Johnson
- [ ] Power transformations

#### 4.3.5 Encoding Categorical Variables
- [ ] Nominal vs ordinal variables
- [ ] Label encoding
- [ ] One-hot encoding
- [ ] Dummy encoding
- [ ] Binary encoding
- [ ] Frequency/target encoding
- [ ] Hashing trick
- [ ] Embedding encoding

#### 4.3.6 Feature Scaling and Normalization
- [ ] When to scale
- [ ] Scaling for different algorithms
- [ ] Pipeline integration

#### 4.3.7 Data Integration
- [ ] Combining multiple data sources
- [ ] Entity resolution
- [ ] Record linkage
- [ ] Schema integration
- [ ] Handling redundancy
- [ ] Detecting and resolving conflicts

#### 4.3.8 Data Reduction
- [ ] Dimensionality reduction overview
- [ ] Feature selection vs extraction
- [ ] Sampling techniques: Random, Stratified, Cluster, Reservoir
- [ ] Data compression basics

---

### 4.4 ETL Concepts (1 week)

#### 4.4.1 ETL Fundamentals
- [ ] What is ETL?
- [ ] Extract, Transform, Load phases
- [ ] ETL vs ELT
- [ ] Batch vs streaming ETL
- [ ] ETL architecture patterns

#### 4.4.2 Extraction
- [ ] Data source identification
- [ ] Full extraction vs incremental extraction
- [ ] Change Data Capture (CDC)
- [ ] API extraction
- [ ] Web scraping basics
- [ ] File extraction (CSV, JSON, XML, Excel)
- [ ] Database extraction

#### 4.4.3 Transformation
- [ ] Data cleansing
- [ ] Data validation
- [ ] Data standardization
- [ ] Data enrichment
- [ ] Aggregation
- [ ] Pivoting/unpivoting
- [ ] Business rule application
- [ ] Data type conversion
- [ ] String manipulation
- [ ] Date/time handling

#### 4.4.4 Loading
- [ ] Full load vs incremental load
- [ ] Upsert operations
- [ ] Load strategies: append, overwrite, merge
- [ ] Bulk loading
- [ ] Transaction management
- [ ] Error handling and recovery
- [ ] Data warehouse concepts

#### 4.4.5 ETL Tools and Technologies
- [ ] Apache Airflow
- [ ] Apache NiFi
- [ ] Talend
- [ ] Informatica
- [ ] dbt (data build tool)
- [ ] AWS Glue
- [ ] Azure Data Factory
- [ ] Google Cloud Dataflow

---

### 4.5 Data Pipelines (1 week)

#### 4.5.1 Pipeline Architecture
- [ ] Pipeline components
- [ ] Pipeline patterns
- [ ] DAGs (Directed Acyclic Graphs)
- [ ] Orchestration
- [ ] Scheduling
- [ ] Dependency management

#### 4.5.2 Pipeline Design
- [ ] Idempotency
- [ ] Fault tolerance
- [ ] Error handling
- [ ] Logging and monitoring
- [ ] Data lineage
- [ ] Versioning

#### 4.5.3 Pipeline Tools
- [ ] Apache Airflow (DAGs, operators, sensors)
- [ ] Prefect
- [ ] Dagster
- [ ] Luigi
- [ ] Kedro
- [ ] MLflow Pipelines

#### 4.5.4 Data Quality in Pipelines
- [ ] Data validation frameworks
- [ ] Great Expectations
- [ ] Deequ
- [ ] Schema enforcement
- [ ] Data testing

#### 4.5.5 Modern Data Stack
- [ ] Data lakes vs data warehouses
- [ ] Data lakehouses
- [ ] Delta Lake
- [ ] Apache Iceberg
- [ ] Apache Hudi
- [ ] Streaming pipelines (Kafka, Spark Streaming)

---

## 5. DSA for AI/ML Engineers (6-8 weeks)

**Prerequisites:** Python programming, discrete mathematics basics

### 5.1 Complexity Analysis (1 week)

#### 5.1.1 Time Complexity
- [ ] What is time complexity?
- [ ] Big O notation
- [ ] Omega (Ω) notation
- [ ] Theta (Θ) notation
- [ ] Common complexities: O(1), O(log n), O(n), O(n log n), O(n²), O(n³), O(2^n), O(n!)
- [ ] Analyzing loops
- [ ] Analyzing recursive algorithms
- [ ] Master theorem
- [ ] Amortized analysis

#### 5.1.2 Space Complexity
- [ ] What is space complexity?
- [ ] Auxiliary space
- [ ] Input space
- [ ] Space-time tradeoff
- [ ] Analyzing recursive space

#### 5.1.3 Complexity Analysis Practice
- [ ] Analyzing common algorithms
- [ ] Best, average, worst case
- [ ] Practical performance considerations

---

### 5.2 Arrays and Strings (1 week)

#### 5.2.1 Arrays
- [ ] Array representation in memory
- [ ] Dynamic arrays
- [ ] Array operations complexity
- [ ] Two-pointer technique
- [ ] Sliding window technique
- [ ] Prefix sum arrays
- [ ] Difference arrays
- [ ] Array rotation
- [ ] Kadane's algorithm (maximum subarray)
- [ ] Dutch national flag problem
- [ ] Trap rain water problem
- [ ] Container with most water
- [ ] Product of array except self
- [ ] Subarray problems
- [ ] Matrix operations: Matrix traversal, Matrix rotation, Spiral traversal, Diagonal traversal

#### 5.2.2 Strings
- [ ] String representation
- [ ] String immutability (Python)
- [ ] String matching algorithms: Naive, Rabin-Karp, KMP, Boyer-Moore, Z-algorithm
- [ ] Palindrome problems
- [ ] Anagram problems
- [ ] Longest common substring/subsequence
- [ ] Edit distance (Levenshtein)
- [ ] String compression
- [ ] Pattern matching
- [ ] Regular expressions basics
- [ ] Trie for strings

---

### 5.3 Linked Lists (1 week)

#### 5.3.1 Singly Linked Lists
- [ ] Node structure
- [ ] Basic operations: insert, delete, search
- [ ] Traversal
- [ ] Head and tail pointers
- [ ] Length calculation

#### 5.3.2 Doubly Linked Lists
- [ ] Node with prev pointer
- [ ] Operations comparison with singly linked
- [ ] Memory overhead

#### 5.3.3 Linked List Problems
- [ ] Reverse a linked list
- [ ] Detect cycle (Floyd's cycle detection)
- [ ] Find cycle start
- [ ] Middle of linked list
- [ ] Nth node from end
- [ ] Intersection of two lists
- [ ] Merge two sorted lists
- [ ] Remove duplicates
- [ ] Palindrome check
- [ ] Copy list with random pointer
- [ ] LRU Cache implementation
- [ ] Fast and slow pointer technique

---

### 5.4 Stacks and Queues (1 week)

#### 5.4.1 Stacks
- [ ] Stack ADT
- [ ] Array implementation
- [ ] Linked list implementation
- [ ] Operations: push, pop, peek
- [ ] Time complexity
- [ ] Applications: Function call stack, Expression evaluation, Backtracking, Undo mechanisms

#### 5.4.2 Stack Problems
- [ ] Valid parentheses
- [ ] Next greater element
- [ ] Previous greater element
- [ ] Largest rectangle in histogram
- [ ] Maximal rectangle
- [ ] Min stack
- [ ] Stock span problem
- [ ] Celebrity problem
- [ ] Infix to postfix/prefix
- [ ] Expression evaluation

#### 5.4.3 Queues
- [ ] Queue ADT
- [ ] Array implementation (circular queue)
- [ ] Linked list implementation
- [ ] Operations: enqueue, dequeue, front, rear
- [ ] Time complexity

#### 5.4.4 Queue Variants
- [ ] Deque (double-ended queue)
- [ ] Priority queue
- [ ] Monotonic queue
- [ ] Circular buffer

#### 5.4.5 Queue Problems
- [ ] Queue using stacks
- [ ] Stack using queues
- [ ] Sliding window maximum
- [ ] First non-repeating character
- [ ] Level order traversal
- [ ] BFS applications

---

### 5.5 Hash Tables (1 week)

#### 5.5.1 Hash Table Fundamentals
- [ ] Hash function properties
- [ ] Collision resolution: Chaining, Open addressing, Linear probing, Quadratic probing, Double hashing
- [ ] Load factor
- [ ] Rehashing
- [ ] Time complexity analysis

#### 5.5.2 Hash Table Applications
- [ ] Frequency counting
- [ ] Two-sum problem
- [ ] Subarray sum equals k
- [ ] Group anagrams
- [ ] Longest consecutive sequence
- [ ] Contains duplicate
- [ ] Valid sudoku
- [ ] Set implementations
- [ ] Dictionary implementations
- [ ] Caching (memoization)

#### 5.5.3 Advanced Hash Topics
- [ ] Rolling hash
- [ ] Rabin-Karp algorithm
- [ ] Bloom filters
- [ ] Consistent hashing
- [ ] Hash trees (Merkle trees)

---

### 5.6 Trees and BSTs (1.5 weeks)

#### 5.6.1 Tree Fundamentals
- [ ] Tree terminology: root, leaf, parent, child, sibling, ancestor, descendant
- [ ] Tree properties
- [ ] Tree representations
- [ ] Binary trees
- [ ] Tree traversals: Preorder, Inorder, Postorder, Level order, Zigzag traversal
- [ ] Height and depth
- [ ] Diameter of tree

#### 5.6.2 Binary Search Trees
- [ ] BST properties
- [ ] Search in BST
- [ ] Insert in BST
- [ ] Delete in BST
- [ ] Inorder successor/predecessor
- [ ] Validate BST
- [ ] Lowest common ancestor
- [ ] Range queries
- [ ] Balanced vs unbalanced BSTs

#### 5.6.3 Balanced Trees
- [ ] AVL trees (rotations)
- [ ] Red-Black trees (concepts)
- [ ] B-trees (concepts)
- [ ] Self-balancing importance

#### 5.6.4 Heaps (Priority Queues)
- [ ] Binary heap properties
- [ ] Min heap vs max heap
- [ ] Heap operations: insert, extract-min/max, peek
- [ ] Heapify
- [ ] Build heap
- [ ] Heap sort
- [ ] Priority queue applications
- [ ] k-way merge
- [ ] Top k elements
- [ ] Median finder
- [ ] Task scheduling

#### 5.6.5 Tree Problems
- [ ] Maximum depth
- [ ] Minimum depth
- [ ] Same tree
- [ ] Symmetric tree
- [ ] Path sum problems
- [ ] Binary tree maximum path sum
- [ ] Serialize/deserialize tree
- [ ] Construct tree from traversals
- [ ] Views of binary tree (left, right, top, bottom)
- [ ] Vertical order traversal
- [ ] Boundary traversal
- [ ] Morris traversal (threaded binary tree)

#### 5.6.6 Trie (Prefix Tree)
- [ ] Trie structure
- [ ] Insert, search, startsWith
- [ ] Delete from trie
- [ ] Applications: Autocomplete, Spell checker, IP routing, Word break, Implement trie, Add and search word, Word search II

#### 5.6.7 Segment Trees and Fenwick Trees
- [ ] Range query problems
- [ ] Segment tree construction
- [ ] Range sum query
- [ ] Range minimum query
- [ ] Lazy propagation
- [ ] Fenwick tree (Binary Indexed Tree)
- [ ] Update and query operations

---

### 5.7 Graphs (1.5 weeks)

#### 5.7.1 Graph Fundamentals
- [ ] Graph terminology: vertices, edges, degree, path, cycle
- [ ] Graph types: directed, undirected, weighted, unweighted
- [ ] Graph representations: Adjacency matrix, Adjacency list, Edge list, Incidence matrix
- [ ] Graph properties

#### 5.7.2 Graph Traversals
- [ ] Breadth-First Search (BFS)
- [ ] Depth-First Search (DFS)

#### 5.7.3 Graph Problems - Connectivity
- [ ] Connected components
- [ ] Strongly connected components (Kosaraju's, Tarjan's)
- [ ] Articulation points
- [ ] Bridges
- [ ] Biconnected components

#### 5.7.4 Graph Problems - Shortest Path
- [ ] Dijkstra's algorithm
- [ ] Bellman-Ford algorithm
- [ ] Floyd-Warshall algorithm
- [ ] A* search algorithm
- [ ] BFS for unweighted graphs
- [ ] All-pairs shortest path

#### 5.7.5 Graph Problems - Minimum Spanning Tree
- [ ] Kruskal's algorithm
- [ ] Prim's algorithm
- [ ] Union-Find (Disjoint Set Union): Find, Union, Path compression, Union by rank/size

#### 5.7.6 Graph Problems - Topological Sort
- [ ] Topological sort using DFS
- [ ] Topological sort using Kahn's algorithm (BFS)
- [ ] Applications: task scheduling, build systems, course schedule

#### 5.7.7 Graph Problems - Cycle Detection
- [ ] Cycle detection in undirected graphs
- [ ] Cycle detection in directed graphs
- [ ] Bipartite graph check
- [ ] Graph coloring

#### 5.7.8 Advanced Graph Topics
- [ ] Eulerian path and circuit
- [ ] Hamiltonian path and circuit
- [ ] Network flow (Ford-Fulkerson)
- [ ] Maximum bipartite matching
- [ ] Graph coloring algorithms
- [ ] Traveling Salesman Problem (TSP)
- [ ] Graph algorithms complexity comparison

---

### 5.8 Sorting and Searching (1 week)

#### 5.8.1 Searching Algorithms
- [ ] Linear search
- [ ] Binary search (Iterative, Recursive, Variants)
- [ ] Ternary search
- [ ] Exponential search
- [ ] Interpolation search
- [ ] Jump search

#### 5.8.2 Sorting Algorithms
- [ ] Bubble sort
- [ ] Selection sort
- [ ] Insertion sort
- [ ] Merge sort (Recursive, Iterative, Time/space complexity, Stability)
- [ ] Quick sort (Lomuto partition, Hoare partition, Randomized, Time/space complexity, Worst case)
- [ ] Heap sort
- [ ] Counting sort
- [ ] Radix sort
- [ ] Bucket sort
- [ ] Tim sort (Python's sort)
- [ ] Comparison of sorting algorithms
- [ ] Stability in sorting
- [ ] In-place sorting

#### 5.8.3 Sorting Applications
- [ ] Merge intervals
- [ ] Meeting rooms problems
- [ ] Activity selection
- [ ] Fractional knapsack
- [ ] H-index
- [ ] Wiggle sort
- [ ] Sort colors
- [ ] Kth largest/smallest element

---

### 5.9 Dynamic Programming (1.5 weeks)

#### 5.9.1 DP Fundamentals
- [ ] What is dynamic programming?
- [ ] Overlapping subproblems
- [ ] Optimal substructure
- [ ] Memoization (top-down)
- [ ] Tabulation (bottom-up)
- [ ] State definition
- [ ] State transitions
- [ ] Base cases
- [ ] Time and space complexity

#### 5.9.2 Classic DP Problems
- [ ] Fibonacci numbers
- [ ] Climbing stairs
- [ ] House robber
- [ ] Coin change
- [ ] Minimum coin change
- [ ] Longest common subsequence (LCS)
- [ ] Longest increasing subsequence (LIS)
- [ ] Edit distance
- [ ] 0/1 Knapsack
- [ ] Unbounded knapsack
- [ ] Subset sum
- [ ] Partition equal subset sum
- [ ] Target sum
- [ ] Rod cutting
- [ ] Matrix chain multiplication
- [ ] Palindromic subsequences
- [ ] Word break
- [ ] Decode ways
- [ ] Unique paths
- [ ] Minimum path sum

#### 5.9.3 DP Patterns
- [ ] Linear DP
- [ ] String DP
- [ ] Interval DP
- [ ] Tree DP
- [ ] DP on grids
- [ ] DP with bitmask
- [ ] Digit DP
- [ ] Probability DP

#### 5.9.4 Optimization Techniques
- [ ] Space optimization in DP
- [ ] State compression
- [ ] Knuth optimization
- [ ] Convex hull trick
- [ ] Divide and conquer optimization

---

### 5.10 Algorithm Design Patterns (1 week)

#### 5.10.1 Greedy Algorithms
- [ ] Greedy choice property
- [ ] Optimal substructure
- [ ] Activity selection
- [ ] Fractional knapsack
- [ ] Huffman coding
- [ ] Job sequencing
- [ ] Minimum number of coins
- [ ] Gas station problem

#### 5.10.2 Divide and Conquer
- [ ] Merge sort
- [ ] Quick sort
- [ ] Binary search
- [ ] Strassen's matrix multiplication
- [ ] Closest pair of points
- [ ] Karatsuba multiplication

#### 5.10.3 Backtracking
- [ ] N-Queens problem
- [ ] Sudoku solver
- [ ] Permutations
- [ ] Combinations
- [ ] Subset problems
- [ ] Word search
- [ ] M-coloring problem
- [ ] Hamiltonian path
- [ ] Rat in a maze

#### 5.10.4 Bit Manipulation
- [ ] Bitwise operators review
- [ ] Common bit tricks
- [ ] Count set bits
- [ ] Check power of 2
- [ ] Single number problems
- [ ] Bit manipulation for subsets
- [ ] XOR properties
- [ ] Mask techniques

#### 5.10.5 Recursion Patterns
- [ ] Understanding recursion
- [ ] Recursion tree
- [ ] Tail recursion
- [ ] Head recursion
- [ ] Tree recursion
- [ ] Indirect recursion
- [ ] Converting recursion to iteration

---

# PHASE 3: CORE ML & DEEP LEARNING (Months 8-12)

## 6. Machine Learning Fundamentals (4-5 weeks)

**Prerequisites:** Python, NumPy, Pandas, Mathematics foundation

### 6.1 Introduction to Machine Learning (1 week)

#### 6.1.1 ML Fundamentals
- [ ] What is Machine Learning?
- [ ] AI vs ML vs Deep Learning
- [ ] Traditional programming vs ML
- [ ] Types of learning: Supervised, Unsupervised, Semi-supervised, Self-supervised, Reinforcement
- [ ] ML workflow overview
- [ ] Applications of ML

#### 6.1.2 ML Problem Types
- [ ] Regression problems
- [ ] Classification problems (Binary, Multi-class, Multi-label)
- [ ] Clustering problems
- [ ] Dimensionality reduction
- [ ] Ranking problems
- [ ] Anomaly detection

#### 6.1.3 Data Preparation for ML
- [ ] Train/Test split
- [ ] Train/Validation/Test split
- [ ] Cross-validation overview
- [ ] Data leakage prevention
- [ ] Feature-target separation

---

### 6.2 Supervised Learning (1.5 weeks)

#### 6.2.1 Regression
- [ ] Simple linear regression
- [ ] Multiple linear regression
- [ ] Polynomial regression
- [ ] Assumptions of linear regression
- [ ] Residual analysis
- [ ] Regression metrics: MAE, MSE, RMSE, R², Adjusted R²

#### 6.2.2 Classification
- [ ] Logistic regression
- [ ] Decision boundaries
- [ ] Probability estimation
- [ ] Classification metrics: Accuracy, Precision, Recall, Specificity, F1-score, Precision-Recall curve, ROC curve, AUC-ROC, Confusion matrix, Matthews Correlation Coefficient, Cohen's Kappa
- [ ] Multi-class metrics: Macro average, Micro average, Weighted average

---

### 6.3 Unsupervised Learning (1 week)

#### 6.3.1 Clustering
- [ ] Clustering concepts
- [ ] Distance metrics: Euclidean, Manhattan, Minkowski, Cosine similarity, Mahalanobis
- [ ] Clustering evaluation: Silhouette score, Davies-Bouldin index, Calinski-Harabasz index, Inertia

#### 6.3.2 Dimensionality Reduction
- [ ] Curse of dimensionality
- [ ] Dimensionality reduction techniques
- [ ] Feature selection vs feature extraction
- [ ] Evaluation of dimensionality reduction

---

### 6.4 Model Evaluation and Validation (1 week)

#### 6.4.1 Cross-Validation
- [ ] Hold-out validation
- [ ] k-fold cross-validation
- [ ] Stratified k-fold
- [ ] Leave-one-out (LOOCV)
- [ ] Leave-p-out
- [ ] Repeated k-fold
- [ ] Time series cross-validation
- [ ] Nested cross-validation
- [ ] Cross-validation for hyperparameter tuning

#### 6.4.2 Bias-Variance Tradeoff
- [ ] Bias error
- [ ] Variance error
- [ ] Irreducible error
- [ ] Bias-variance decomposition
- [ ] Underfitting
- [ ] Overfitting
- [ ] Learning curves
- [ ] Validation curves
- [ ] Regularization concepts

#### 6.4.3 Model Selection
- [ ] Model comparison
- [ ] Statistical tests for model comparison
- [ ] Model combination: Voting, Averaging, Stacking, Blending

---

### 6.5 Feature Engineering (1 week)

#### 6.5.1 Feature Selection
- [ ] Filter methods: Variance threshold, Correlation-based, Chi-square test, Mutual information, ANOVA F-test
- [ ] Wrapper methods: Forward selection, Backward elimination, Recursive Feature Elimination (RFE)
- [ ] Embedded methods: Lasso (L1) regularization, Tree-based importance, Elastic Net

#### 6.5.2 Feature Extraction
- [ ] PCA (Principal Component Analysis)
- [ ] LDA (Linear Discriminant Analysis)
- [ ] Kernel PCA
- [ ] t-SNE (visualization)
- [ ] UMAP (visualization)
- [ ] Autoencoders

#### 6.5.3 Feature Creation
- [ ] Domain-based features
- [ ] Polynomial features
- [ ] Interaction features
- [ ] Binning/discretization
- [ ] Date/time features
- [ ] Text features
- [ ] Aggregation features

#### 6.5.4 Feature Transformation
- [ ] Scaling (review)
- [ ] Normalization (review)
- [ ] Power transforms
- [ ] Quantile transformation
- [ ] Yeo-Johnson
- [ ] Box-Cox

---

## 7. Classical ML Algorithms (6-8 weeks)

**Prerequisites:** ML Fundamentals, Python, scikit-learn basics

### 7.1 Linear Models (1.5 weeks)

#### 7.1.1 Linear Regression
- [ ] Ordinary Least Squares (OLS)
- [ ] Closed-form solution (Normal equation)
- [ ] Gradient descent for linear regression
- [ ] Assumptions and diagnostics
- [ ] Multicollinearity
- [ ] VIF (Variance Inflation Factor)
- [ ] Regularized linear regression: Ridge (L2), Lasso (L1), Elastic Net
- [ ] Polynomial regression
- [ ] Generalized Linear Models (GLM)

#### 7.1.2 Logistic Regression
- [ ] Sigmoid function
- [ ] Decision boundary
- [ ] Maximum Likelihood Estimation
- [ ] Cost function (cross-entropy)
- [ ] Gradient descent
- [ ] Multi-class logistic regression: One-vs-Rest (OvR), One-vs-One (OvO), Softmax regression
- [ ] Regularization in logistic regression
- [ ] Class imbalance handling: Class weights, Sampling techniques

#### 7.1.3 Generalized Linear Models
- [ ] Exponential family
- [ ] Link functions
- [ ] Poisson regression
- [ ] Gamma regression

---

### 7.2 Tree-Based Models (2 weeks)

#### 7.2.1 Decision Trees
- [ ] Tree structure
- [ ] Splitting criteria: Gini impurity, Entropy and Information Gain, Chi-square, Variance reduction
- [ ] Tree building algorithm
- [ ] Pruning: Pre-pruning, Post-pruning
- [ ] Handling continuous features
- [ ] Handling missing values
- [ ] Feature importance
- [ ] Tree visualization
- [ ] Advantages and limitations

#### 7.2.2 Ensemble Methods - Bagging
- [ ] Bootstrap sampling
- [ ] Bagging concept
- [ ] Random Forest: Algorithm, Feature subsampling, Out-of-bag (OOB) error, Feature importance, Hyperparameters
- [ ] Extra Trees (Extremely Randomized Trees)

#### 7.2.3 Ensemble Methods - Boosting
- [ ] Boosting concept
- [ ] AdaBoost: Algorithm, Sample weighting, Classifier weighting
- [ ] Gradient Boosting: Gradient descent in function space, Loss functions, Learning rate, Number of estimators
- [ ] XGBoost: Regularization, Handling missing values, Parallel processing, Hyperparameters, Early stopping
- [ ] LightGBM: GOSS, EFB, Leaf-wise growth, Hyperparameters
- [ ] CatBoost: Ordered boosting, Categorical feature handling, Symmetric trees, Hyperparameters

#### 7.2.4 Ensemble Comparison and Best Practices
- [ ] When to use which ensemble
- [ ] Hyperparameter tuning for ensembles
- [ ] Feature importance interpretation
- [ ] Partial dependence plots
- [ ] SHAP values introduction

---

### 7.3 Instance-Based Learning (1 week)

#### 7.3.1 K-Nearest Neighbors (KNN)
- [ ] KNN algorithm
- [ ] Distance metrics
- [ ] Choosing k
- [ ] Weighted KNN
- [ ] KD-trees for efficient search
- [ ] Ball trees
- [ ] Time and space complexity
- [ ] Applications

#### 7.3.2 Learning Vector Quantization (LVQ)
- [ ] Codebook vectors
- [ ] Training algorithm
- [ ] Variants

---

### 7.4 Support Vector Machines (1 week)

#### 7.4.1 SVM Fundamentals
- [ ] Maximum margin classifier
- [ ] Support vectors
- [ ] Hard margin SVM
- [ ] Soft margin SVM
- [ ] Hinge loss
- [ ] Primal and dual formulation
- [ ] Kernel trick

#### 7.4.2 Kernel Functions
- [ ] Linear kernel
- [ ] Polynomial kernel
- [ ] RBF (Gaussian) kernel
- [ ] Sigmoid kernel
- [ ] Custom kernels
- [ ] Choosing kernels

#### 7.4.3 SVM for Regression
- [ ] Support Vector Regression (SVR)
- [ ] Epsilon-insensitive loss
- [ ] Parameters: C, epsilon, kernel

#### 7.4.4 SVM Implementation
- [ ] SMO algorithm (conceptual)
- [ ] Hyperparameter tuning
- [ ] Multi-class SVM
- [ ] Probability estimates
- [ ] Scalability issues

---

### 7.5 Naive Bayes (1 week)

#### 7.5.1 Bayes Theorem Review
- [ ] Conditional probability
- [ ] Bayes theorem
- [ ] Prior, likelihood, posterior

#### 7.5.2 Naive Bayes Variants
- [ ] Gaussian Naive Bayes
- [ ] Multinomial Naive Bayes
- [ ] Bernoulli Naive Bayes
- [ ] Complement Naive Bayes
- [ ] Categorical Naive Bayes

#### 7.5.3 Naive Bayes Applications
- [ ] Text classification
- [ ] Spam filtering
- [ ] Sentiment analysis
- [ ] Laplace smoothing
- [ ] Limitations and assumptions

---

### 7.6 Clustering Algorithms (1.5 weeks)

#### 7.6.1 K-Means
- [ ] Algorithm
- [ ] Initialization methods: Random, K-Means++
- [ ] Choosing k: Elbow method, Silhouette analysis, Gap statistic
- [ ] Limitations
- [ ] Variants: K-Medoids (PAM), K-Medians, Fuzzy C-Means, Mini-batch K-Means

#### 7.6.2 DBSCAN
- [ ] Density-based clustering
- [ ] Core points, border points, noise
- [ ] Parameters: eps, min_samples
- [ ] Advantages over K-Means
- [ ] HDBSCAN

#### 7.6.3 Hierarchical Clustering
- [ ] Agglomerative clustering
- [ ] Divisive clustering
- [ ] Dendrograms
- [ ] Linkage criteria: Single, Complete, Average, Ward
- [ ] Choosing number of clusters
- [ ] Cophenetic correlation

#### 7.6.4 Other Clustering Methods
- [ ] Gaussian Mixture Models (GMM)
- [ ] Expectation-Maximization (EM) algorithm
- [ ] Spectral clustering
- [ ] Affinity propagation
- [ ] Mean shift
- [ ] OPTICS
- [ ] BIRCH

---

### 7.7 Dimensionality Reduction (1 week)

#### 7.7.1 PCA (Principal Component Analysis)
- [ ] Variance maximization
- [ ] Eigendecomposition approach
- [ ] SVD approach
- [ ] Choosing number of components
- [ ] Explained variance ratio
- [ ] PCA assumptions
- [ ] Kernel PCA

#### 7.7.2 LDA (Linear Discriminant Analysis)
- [ ] Class separability
- [ ] Scatter matrices
- [ ] Fisher's criterion
- [ ] Multi-class LDA
- [ ] PCA vs LDA

#### 7.7.3 Manifold Learning
- [ ] t-SNE: Algorithm, Perplexity parameter, Visualization use
- [ ] UMAP: Algorithm, Advantages over t-SNE, Parameters
- [ ] Isomap
- [ ] Locally Linear Embedding (LLE)
- [ ] MDS (Multidimensional Scaling)

#### 7.7.4 Autoencoders for Dimensionality Reduction
- [ ] Encoder-decoder architecture
- [ ] Bottleneck layer
- [ ] Variational Autoencoders (VAE)

---

## 8. Deep Learning Fundamentals (6-8 weeks)

**Prerequisites:** Classical ML, Calculus, Linear Algebra, Python

### 8.1 Neural Network Basics (2 weeks)

#### 8.1.1 Biological Inspiration
- [ ] Biological neurons
- [ ] Artificial neurons
- [ ] History of neural networks

#### 8.1.2 Perceptron
- [ ] Perceptron model
- [ ] Perceptron learning algorithm
- [ ] Perceptron convergence
- [ ] Limitations (XOR problem)

#### 8.1.3 Multi-Layer Perceptron (MLP)
- [ ] Architecture: input, hidden, output layers
- [ ] Forward propagation
- [ ] Universal approximation theorem
- [ ] Network depth vs width

#### 8.1.4 Activation Functions
- [ ] Step function
- [ ] Sigmoid
- [ ] Tanh
- [ ] ReLU (Rectified Linear Unit)
- [ ] Leaky ReLU
- [ ] Parametric ReLU (PReLU)
- [ ] Exponential Linear Unit (ELU)
- [ ] GELU (Gaussian Error Linear Unit)
- [ ] Swish
- [ ] Softmax
- [ ] Comparison and selection

#### 8.1.5 Loss Functions
- [ ] Regression losses: MSE, MAE, Huber loss, Log-Cosh loss
- [ ] Classification losses: Binary cross-entropy, Categorical cross-entropy, Sparse categorical cross-entropy, Hinge loss, Focal loss
- [ ] Custom loss functions

#### 8.1.6 Backpropagation
- [ ] Chain rule review
- [ ] Computational graphs
- [ ] Gradient computation
- [ ] Backpropagation algorithm
- [ ] Vectorized implementation
- [ ] Numerical gradient checking

---

### 8.2 Optimization Algorithms (1.5 weeks)

#### 8.2.1 Gradient Descent Variants
- [ ] Batch Gradient Descent
- [ ] Stochastic Gradient Descent (SGD)
- [ ] Mini-batch Gradient Descent
- [ ] Learning rate considerations

#### 8.2.2 Momentum-Based Methods
- [ ] Momentum
- [ ] Nesterov Accelerated Gradient (NAG)
- [ ] Physical interpretation

#### 8.2.3 Adaptive Learning Rate Methods
- [ ] AdaGrad
- [ ] RMSprop
- [ ] Adam
- [ ] AdaDelta
- [ ] AdamW (Adam with weight decay)
- [ ] Nadam
- [ ] Comparison and recommendations

#### 8.2.4 Learning Rate Scheduling
- [ ] Step decay
- [ ] Exponential decay
- [ ] Cosine annealing
- [ ] Warm restarts
- [ ] Learning rate warmup
- [ ] ReduceLROnPlateau
- [ ] Cyclical learning rates
- [ ] One-cycle policy

#### 8.2.5 Advanced Optimization Topics
- [ ] Gradient clipping
- [ ] Gradient accumulation
- [ ] Second-order methods (conceptual): Newton's method, L-BFGS
- [ ] Optimization landscape visualization

---

### 8.3 Regularization Techniques (1 week)

#### 8.3.1 L1 and L2 Regularization
- [ ] L2 (Ridge) regularization
- [ ] L1 (Lasso) regularization
- [ ] Elastic Net
- [ ] Weight decay

#### 8.3.2 Dropout
- [ ] Dropout algorithm
- [ ] Inverted dropout
- [ ] Dropout rate selection
- [ ] Variants: Spatial dropout, DropConnect, Variational dropout

#### 8.3.3 Batch Normalization
- [ ] Internal covariate shift
- [ ] BatchNorm algorithm
- [ ] Training vs inference
- [ ] Gamma and beta parameters
- [ ] Pros and cons
- [ ] Layer Normalization
- [ ] Instance Normalization
- [ ] Group Normalization

#### 8.3.4 Other Regularization Methods
- [ ] Early stopping
- [ ] Data augmentation
- [ ] Label smoothing
- [ ] Mixup
- [ ] Cutout
- [ ] CutMix
- [ ] Stochastic depth
- [ ] Weight constraints

---

### 8.4 Training Deep Networks (1.5 weeks)

#### 8.4.1 Weight Initialization
- [ ] Zero initialization (problem)
- [ ] Random initialization
- [ ] Xavier/Glorot initialization
- [ ] He initialization
- [ ] LeCun initialization
- [ ] Orthogonal initialization

#### 8.4.2 Vanishing/Exploding Gradients
- [ ] Problem identification
- [ ] Solutions: Proper initialization, Batch normalization, Residual connections, Gradient clipping, Activation function choice

#### 8.4.3 Batch Training
- [ ] Batch size selection
- [ ] Memory considerations
- [ ] Gradient accumulation
- [ ] Mixed precision training

#### 8.4.4 Debugging Neural Networks
- [ ] Common issues
- [ ] Debugging checklist
- [ ] Visualization tools
- [ ] Gradient flow analysis

---

### 8.5 Convolutional Neural Networks (2 weeks)

#### 8.5.1 CNN Fundamentals
- [ ] Why CNNs for images?
- [ ] Convolution operation
- [ ] Kernel/filter
- [ ] Feature maps
- [ ] Stride
- [ ] Padding: Valid padding, Same padding
- [ ] Receptive field

#### 8.5.2 CNN Layers
- [ ] Convolutional layer
- [ ] Pooling layers: Max pooling, Average pooling, Global pooling
- [ ] Fully connected layer
- [ ] Flatten layer
- [ ] Dropout layer
- [ ] BatchNorm layer

#### 8.5.3 CNN Architectures
- [ ] LeNet-5
- [ ] AlexNet
- [ ] VGG (VGG16, VGG19)
- [ ] Network in Network
- [ ] GoogLeNet / Inception (Inception modules, Auxiliary classifiers)
- [ ] ResNet (Residual blocks, Identity connections, ResNet variants)
- [ ] DenseNet (Dense blocks, Feature reuse)
- [ ] MobileNet (Depthwise separable convolutions)
- [ ] EfficientNet (Compound scaling)
- [ ] Xception
- [ ] SqueezeNet

#### 8.5.4 CNN Techniques
- [ ] Transfer learning
- [ ] Fine-tuning
- [ ] Feature extraction
- [ ] Data augmentation for CNNs
- [ ] Visualization: Filter visualization, Activation maps, Grad-CAM, Saliency maps

#### 8.5.5 CNN Applications
- [ ] Image classification
- [ ] Object detection (intro)
- [ ] Semantic segmentation (intro)
- [ ] Image retrieval
- [ ] Style transfer

---

### 8.6 Recurrent Neural Networks (1.5 weeks)

#### 8.6.1 Sequence Modeling
- [ ] Sequence data types
- [ ] Challenges with sequences
- [ ] Traditional approaches limitations

#### 8.6.2 RNN Fundamentals
- [ ] RNN architecture
- [ ] Unrolling through time
- [ ] Hidden state
- [ ] Forward propagation
- [ ] Backpropagation Through Time (BPTT)
- [ ] Vanishing gradient problem in RNNs

#### 8.6.3 LSTM (Long Short-Term Memory)
- [ ] LSTM cell structure
- [ ] Forget gate
- [ ] Input gate
- [ ] Output gate
- [ ] Cell state
- [ ] LSTM variants
- [ ] Peephole connections

#### 8.6.4 GRU (Gated Recurrent Unit)
- [ ] GRU cell structure
- [ ] Reset gate
- [ ] Update gate
- [ ] GRU vs LSTM comparison

#### 8.6.5 Bidirectional RNNs
- [ ] Bidirectional architecture
- [ ] Applications
- [ ] Combining with LSTM/GRU

#### 8.6.6 Deep RNNs
- [ ] Stacking RNN layers
- [ ] Skip connections

#### 8.6.7 RNN Applications
- [ ] Time series prediction
- [ ] Sequence classification
- [ ] Many-to-many problems
- [ ] Sequence generation

---

## 9. Deep Learning Frameworks (4-5 weeks)

**Prerequisites:** Deep Learning Fundamentals, Python

### 9.1 TensorFlow/Keras (2 weeks)

#### 9.1.1 TensorFlow Basics
- [ ] TensorFlow architecture
- [ ] Tensors
- [ ] Operations
- [ ] Variables
- [ ] GradientTape
- [ ] Eager execution
- [ ] tf.function and graph mode

#### 9.1.2 Keras API
- [ ] Sequential API
- [ ] Functional API
- [ ] Model subclassing
- [ ] Layers: Core, Convolutional, Pooling, Recurrent, Embedding, Normalization, Attention
- [ ] Custom layers
- [ ] Custom models

#### 9.1.3 Model Training
- [ ] Compiling models
- [ ] Optimizers
- [ ] Loss functions
- [ ] Metrics
- [ ] Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger, LearningRateScheduler
- [ ] fit(), evaluate(), predict()
- [ ] Training loops

#### 9.1.4 Data Pipeline
- [ ] tf.data API
- [ ] Dataset creation
- [ ] Transformations: map, batch, shuffle, prefetch
- [ ] Performance optimization
- [ ] ImageDataGenerator
- [ ] Keras preprocessing layers

#### 9.1.5 Advanced TensorFlow
- [ ] Distributed training
- [ ] Mixed precision
- [ ] SavedModel format
- [ ] TensorFlow Serving
- [ ] TensorFlow Lite
- [ ] TensorFlow.js
- [ ] Custom training loops

---

### 9.2 PyTorch (2 weeks)

#### 9.2.1 PyTorch Basics
- [ ] Tensors
- [ ] Tensor operations
- [ ] Autograd
- [ ] Computational graphs
- [ ] Device management (CPU/GPU)

#### 9.2.2 Neural Networks with PyTorch
- [ ] nn.Module
- [ ] nn.Sequential
- [ ] Built-in layers
- [ ] Custom layers
- [ ] Custom models
- [ ] Parameter registration

#### 9.2.3 Training in PyTorch
- [ ] Loss functions
- [ ] Optimizers
- [ ] Training loop
- [ ] Validation loop
- [ ] Learning rate scheduling
- [ ] Gradient clipping

#### 9.2.4 Data Loading
- [ ] Dataset class
- [ ] DataLoader
- [ ] Transforms
- [ ] torchvision datasets
- [ ] Custom datasets
- [ ] Distributed sampling

#### 9.2.5 Advanced PyTorch
- [ ] torch.nn.functional
- [ ] Hooks
- [ ] Custom autograd functions
- [ ] torchscript
- [ ] Distributed training
- [ ] AMP (Automatic Mixed Precision)
- [ ] Profiling

---

### 9.3 JAX (Optional, 1 week)

#### 9.3.1 JAX Fundamentals
- [ ] JAX philosophy
- [ ] NumPy compatibility
- [ ] jit compilation
- [ ] grad transformation
- [ ] vmap transformation
- [ ] pmap for parallelism

#### 9.3.2 JAX for Deep Learning
- [ ] Flax
- [ ] Haiku
- [ ] Optax
- [ ] Comparison with PyTorch/TensorFlow

---

# PHASE 4: SPECIALIZATION (Months 13-18)

## 10. Natural Language Processing (8-10 weeks)

**Prerequisites:** Deep Learning, Python, Classical ML

### 10.1 NLP Fundamentals (2 weeks)

#### 10.1.1 Text Preprocessing
- [ ] Tokenization: Word, Sentence, Subword
- [ ] Lowercasing
- [ ] Stopword removal
- [ ] Stemming
- [ ] Lemmatization
- [ ] Part-of-speech tagging
- [ ] Named Entity Recognition (NER)
- [ ] Chunking
- [ ] Dependency parsing

#### 10.1.2 Text Representation
- [ ] One-hot encoding
- [ ] Bag of Words (BoW)
- [ ] N-grams
- [ ] TF-IDF
- [ ] Hashing trick
- [ ] Character-level representations

#### 10.1.3 Word Embeddings
- [ ] Word2Vec: CBOW, Skip-gram, Negative sampling, Hierarchical softmax
- [ ] GloVe (Global Vectors)
- [ ] FastText
- [ ] Embedding matrix
- [ ] Pre-trained embeddings
- [ ] Embedding visualization

---

### 10.2 Sequence Models for NLP (2 weeks)

#### 10.2.1 RNNs for NLP
- [ ] Text classification with RNNs
- [ ] Sequence labeling
- [ ] Named Entity Recognition
- [ ] Sentiment analysis

#### 10.2.2 Encoder-Decoder Architecture
- [ ] Sequence-to-sequence models
- [ ] Encoder
- [ ] Decoder
- [ ] Applications: Machine translation, Text summarization, Question answering

#### 10.2.3 Attention Mechanism
- [ ] Attention motivation
- [ ] Bahdanau attention
- [ ] Luong attention
- [ ] Self-attention
- [ ] Multi-head attention
- [ ] Attention visualization

---

### 10.3 Transformer Architecture (2 weeks)

#### 10.3.1 Transformer Fundamentals
- [ ] Original Transformer paper
- [ ] Architecture components: Multi-head self-attention, Position-wise feed-forward networks, Layer normalization, Residual connections
- [ ] Positional encodings (Sinusoidal vs. RoPE - Rotary Positional Embeddings)
- [ ] Layer normalization (Post-LN vs Pre-LN vs RMSNorm)
- [ ] FlashAttention-1/2/3 optimization logic
- [ ] Mixture of Experts (MoE) architecture (Router, Experts)
- [ ] Scaling Laws (Chinchilla, Kaplan)
- [ ] Encoder stack
- [ ] Decoder stack
- [ ] Masked attention

#### 10.3.2 Transformer Variants
- [ ] Encoder-only (BERT-style)
- [ ] Decoder-only (GPT-style)
- [ ] Encoder-decoder (T5-style)
- [ ] Vision Transformers (ViT)

---

### 10.4 Pre-trained Language Models (2 weeks)

#### 10.4.1 BERT Family
- [ ] BERT architecture
- [ ] Pre-training objectives: MLM, NSP
- [ ] BERT variants: BERT-base, BERT-large, RoBERTa, DistilBERT, ALBERT, ELECTRA
- [ ] Fine-tuning BERT

#### 10.4.2 GPT Family
- [ ] GPT architecture
- [ ] Autoregressive language modeling
- [ ] GPT-2
- [ ] GPT-3
- [ ] GPT-3.5
- [ ] GPT-4
- [ ] Instruction tuning

#### 10.4.3 Other Transformer Models
- [ ] T5 (Text-to-Text Transfer Transformer)
- [ ] XLNet
- [ ] BART
- [ ] DeBERTa
- [ ] Longformer
- [ ] BigBird
- [ ] LLaMA family

---

### 10.5 LLMs and Prompt Engineering (1.5 weeks)

#### 10.5.1 Large Language Models
- [ ] Scale and capabilities
- [ ] Emergent abilities
- [ ] In-context learning
- [ ] Few-shot learning
- [ ] Zero-shot learning

#### 10.5.2 Prompt Engineering
- [ ] Prompt design principles
- [ ] Zero-shot prompting
- [ ] Few-shot prompting
- [ ] Chain-of-thought prompting
- [ ] Self-consistency
- [ ] Tree-of-thoughts
- [ ] Prompt templates
- [ ] System prompts
- [ ] Role prompting

#### 10.5.3 Advanced Prompting
- [ ] Retrieval-augmented generation
- [ ] Tool use with prompts
- [ ] Function calling
- [ ] Structured output
- [ ] Prompt optimization

---

### 10.6 RAG Architectures (1.5 weeks)

#### 10.6.1 RAG Fundamentals
- [ ] Retrieval-Augmented Generation concept
- [ ] Components: Retriever, Generator, Knowledge base
- [ ] Dense retrieval
- [ ] Sparse retrieval
- [ ] Hybrid retrieval

#### 10.6.2 Vector Databases
- [ ] Embedding storage
- [ ] Similarity search
- [ ] Approximate nearest neighbors
- [ ] Popular vector DBs: Pinecone, Milvus, Chroma, Weaviate, Qdrant

#### 10.6.3 RAG Implementation
- [ ] Document chunking
- [ ] Embedding generation
- [ ] Index creation
- [ ] Query processing
- [ ] Context construction
- [ ] Answer generation
- [ ] Evaluation metrics

#### 10.6.4 Advanced RAG
- [ ] Multi-hop retrieval
- [ ] Iterative retrieval
- [ ] Re-ranking (Cohere, BGE-reranker)
- [ ] Query expansion
- [ ] Hypothetical document embeddings (HyDE)
- [ ] GraphRAG (Knowledge Graphs for RAG)
- [ ] Agentic RAG (Router-based and Tool-based retrieval)

---

### 10.7 Fine-tuning Techniques (1.5 weeks)

#### 10.7.1 Full Fine-tuning
- [ ] Standard fine-tuning
- [ ] Layer-wise learning rates
- [ ] Discriminative fine-tuning

#### 10.7.2 Parameter-Efficient Fine-tuning
- [ ] LoRA (Low-Rank Adaptation)
- [ ] QLoRA (Quantized LoRA)
- [ ] Prefix tuning
- [ ] Prompt tuning
- [ ] P-tuning
- [ ] Adapter methods

#### 10.7.3 Alignment & Preference Optimization
- [ ] RLHF (Reinforcement Learning from Human Feedback)
- [ ] Reward Modeling
- [ ] PPO for Language Models
- [ ] DPO (Direct Preference Optimization)
- [ ] KTO / ORPO

#### 10.7.4 Fine-tuning Best Practices
- [ ] Dataset preparation
- [ ] Hyperparameter selection
- [ ] Evaluation
- [ ] Catastrophic forgetting prevention

---

### 10.8 NLP Applications (1 week)

#### 10.8.1 Text Classification
- [ ] Sentiment analysis
- [ ] Topic classification
- [ ] Intent detection
- [ ] Spam detection

#### 10.8.2 Sequence Labeling
- [ ] POS tagging
- [ ] NER
- [ ] Chunking

#### 10.8.3 Text Generation
- [ ] Language modeling
- [ ] Story generation
- [ ] Code generation
- [ ] Dialogue systems

#### 10.8.4 Machine Translation
- [ ] Neural machine translation
- [ ] Multi-lingual models

#### 10.8.5 Question Answering
- [ ] Extractive QA
- [ ] Generative QA
- [ ] Open-domain QA

#### 10.8.6 Text Summarization
- [ ] Extractive summarization
- [ ] Abstractive summarization

---

### 10.9 LLMOps & Evaluation (1.5 weeks)

#### 10.9.1 Evaluation Metrics
- [ ] Traditional vs LLM Metrics
- [ ] Perplexity & Cross-entropy
- [ ] BLEU & ROUGE limitations
- [ ] Human Evaluation

#### 10.9.2 RAG Evaluation
- [ ] Ragas framework
- [ ] TruLens
- [ ] DeepEval
- [ ] Context Precision & Recall
- [ ] Faithfulness & Answer Relevance

#### 10.9.3 LLM Tracing & Observability
- [ ] LangSmith
- [ ] Arize Phoenix
- [ ] Weights & Biases Prompts
- [ ] Prompt injection & defense

---

## 11. Computer Vision (8-10 weeks)

**Prerequisites:** Deep Learning, CNNs, Python

### 11.1 Image Preprocessing (1 week)

#### 11.1.1 Basic Operations
- [ ] Image representation
- [ ] Color spaces: RGB, HSV, LAB, grayscale
- [ ] Resizing
- [ ] Cropping
- [ ] Flipping
- [ ] Rotation
- [ ] Translation

#### 11.1.2 Image Enhancement
- [ ] Histogram equalization
- [ ] Contrast adjustment
- [ ] Brightness adjustment
- [ ] Sharpening
- [ ] Blurring
- [ ] Noise reduction

#### 11.1.3 Advanced Preprocessing
- [ ] Normalization
- [ ] Standardization
- [ ] Color augmentation
- [ ] Geometric transformations
- [ ] Elastic transformations

---

### 11.2 CNN Architectures (2 weeks)

#### 11.2.1 Classic Architectures Review
- [ ] LeNet-5
- [ ] AlexNet
- [ ] VGG
- [ ] GoogLeNet/Inception
- [ ] ResNet
- [ ] DenseNet

#### 11.2.2 Modern Architectures
- [ ] MobileNet (v1, v2, v3)
- [ ] ShuffleNet
- [ ] EfficientNet (B0-B7)
- [ ] RegNet
- [ ] ConvNeXt
- [ ] Vision Transformers (ViT)
- [ ] Swin Transformer

#### 11.2.3 Architecture Selection
- [ ] Accuracy vs efficiency
- [ ] Model size considerations
- [ ] Inference speed
- [ ] Mobile deployment

---

### 11.3 Object Detection (2 weeks)

#### 11.3.1 Detection Fundamentals
- [ ] Bounding boxes
- [ ] IoU (Intersection over Union)
- [ ] NMS (Non-Maximum Suppression)
- [ ] mAP (mean Average Precision)

#### 11.3.2 Two-Stage Detectors
- [ ] R-CNN
- [ ] Fast R-CNN
- [ ] Faster R-CNN (RPN, ROI pooling, ROI align)
- [ ] Mask R-CNN

#### 11.3.3 One-Stage Detectors
- [ ] YOLO family (v1-v8): Architecture evolution, Anchor boxes, Multi-scale prediction
- [ ] SSD (Single Shot Detector)
- [ ] RetinaNet (Focal loss, Feature Pyramid Networks)

#### 11.3.4 Transformer-based Detection
- [ ] DETR
- [ ] Deformable DETR
- [ ] Swin Transformer for detection

---

### 11.4 Image Segmentation (2 weeks)

#### 11.4.1 Semantic Segmentation
- [ ] FCN (Fully Convolutional Networks)
- [ ] U-Net (Encoder-decoder, Skip connections)
- [ ] DeepLab family (Atrous convolution, Atrous Spatial Pyramid Pooling)
- [ ] PSPNet
- [ ] SegNet

#### 11.4.2 Instance Segmentation
- [ ] Mask R-CNN
- [ ] SOLO
- [ ] YOLACT

#### 11.4.3 Panoptic Segmentation
- [ ] Combining semantic and instance
- [ ] Panoptic FPN

---

### 11.5 Image Generation (2 weeks)

#### 11.5.1 Autoencoders
- [ ] Basic autoencoders
- [ ] Denoising autoencoders
- [ ] Variational Autoencoders (VAE): Reparameterization trick, KL divergence, Latent space

#### 11.5.2 GANs (Generative Adversarial Networks)
- [ ] GAN architecture
- [ ] Generator
- [ ] Discriminator
- [ ] Adversarial training
- [ ] DCGAN
- [ ] Conditional GANs
- [ ] CycleGAN
- [ ] StyleGAN
- [ ] StyleGAN2
- [ ] StyleGAN3
- [ ] GAN training challenges
- [ ] Mode collapse

#### 11.5.3 Diffusion Models
- [ ] Diffusion process
- [ ] Forward diffusion
- [ ] Reverse diffusion
- [ ] DDPM (Denoising Diffusion Probabilistic Models)
- [ ] DDIM (Denoising Diffusion Implicit Models)
- [ ] Stable Diffusion (Latent diffusion, Text conditioning, CLIP guidance)
- [ ] ControlNet
- [ ] LoRA for diffusion

#### 11.5.4 Flow-based Models
- [ ] Normalizing flows
- [ ] Change of variables
- [ ] RealNVP
- [ ] Glow
- [ ] NICE

---

### 11.6 Vision Transformers (1.5 weeks)

#### 11.6.1 ViT Fundamentals
- [ ] Patch embeddings
- [ ] Position embeddings
- [ ] Transformer encoder
- [ ] Class token
- [ ] Training strategies

#### 11.6.2 ViT Variants
- [ ] DeiT (Data-efficient Image Transformers)
- [ ] Swin Transformer (Shifted windows, Hierarchical architecture)
- [ ] BEiT
- [ ] MAE (Masked Autoencoders)
- [ ] DINO

#### 11.6.3 Hybrid Architectures
- [ ] Convolutional Vision Transformer
- [ ] MobileViT

---

### 11.7 Advanced CV Topics (1.5 weeks)

#### 11.7.1 Face Recognition
- [ ] Face detection
- [ ] Face alignment
- [ ] Face embeddings
- [ ] ArcFace, CosFace, SphereFace

#### 11.7.2 Pose Estimation
- [ ] 2D pose estimation
- [ ] 3D pose estimation
- [ ] OpenPose
- [ ] HRNet

#### 11.7.3 OCR (Optical Character Recognition)
- [ ] Text detection
- [ ] Text recognition
- [ ] CRNN
- [ ] CRAFT

#### 11.7.4 Video Understanding
- [ ] Video classification
- [ ] Action recognition
- [ ] 3D CNNs
- [ ] Two-stream networks
- [ ] Video transformers

#### 11.7.5 Self-Supervised Learning
- [ ] Contrastive learning
- [ ] SimCLR
- [ ] MoCo
- [ ] BYOL
- [ ] DINO

---

### 11.8 Multimodal AI & Vision-Language Models (1.5 weeks)

#### 11.8.1 Contrastive Learning
- [ ] CLIP (Contrastive Language-Image Pre-training)
- [ ] ALIGN
- [ ] Zero-shot image classification

#### 11.8.2 Vision-Language Models (VLMs)
- [ ] LLaVA architecture
- [ ] Flamingo
- [ ] GPT-4V / Gemini multimodal parsing
- [ ] Cross-modal attention

---

### 11.9 Audio & Speech Processing (1.5 weeks)

#### 11.9.1 Audio Preprocessing
- [ ] Waveforms and spectrograms
- [ ] Mel-Frequency Cepstral Coefficients (MFCCs)
- [ ] Log-Mel spectrograms
- [ ] Audio augmentation

#### 11.9.2 Speech Recognition (ASR)
- [ ] Wav2Vec 2.0
- [ ] Whisper architecture
- [ ] Connectionist Temporal Classification (CTC) loss
- [ ] End-to-end ASR

#### 11.9.3 Speech Generation (TTS)
- [ ] Tacotron 2
- [ ] VALL-E
- [ ] FastSpeech
- [ ] Voice cloning

---

# PHASE 5: PRODUCTION & ADVANCED (Months 19-24)

## 12. MLOps & Production (6-8 weeks)

**Prerequisites:** Deep Learning, Software Engineering basics

### 12.1 Model Deployment (2 weeks)

#### 12.1.1 Deployment Patterns
- [ ] Batch inference
- [ ] Real-time inference
- [ ] Streaming inference
- [ ] Edge deployment

#### 12.1.2 REST APIs & Background Tasks
- [ ] Flask basics
- [ ] FastAPI
- [ ] Async Task Processing (Celery)
- [ ] Redis (as message broker/result backend)
- [ ] API design
- [ ] Request/response handling
- [ ] Error handling
- [ ] Authentication
- [ ] Rate limiting
- [ ] Documentation (OpenAPI/Swagger)

#### 12.1.3 Model Serving
- [ ] TensorFlow Serving
- [ ] TorchServe
- [ ] Triton Inference Server
- [ ] ONNX Runtime
- [ ] Model optimization for serving

#### 12.1.4 Serverless Deployment
- [ ] AWS Lambda
- [ ] Google Cloud Functions
- [ ] Azure Functions
- [ ] Cold start considerations

#### 12.1.5 LLM Inference & Serving
- [ ] vLLM and PagedAttention
- [ ] Text Generation Inference (TGI)
- [ ] Ollama and GGML/GGUF models
- [ ] Local LLMs (llama.cpp)
- [ ] TensorRT-LLM
- [ ] KV-Cache optimization (Continuous Batching, Copy-on-Write)
- [ ] Speculative Decoding (Draft models, Medusa)
- [ ] Multi-Query Attention (MQA) & Grouped-Query Attention (GQA) logic

---

### 12.2 Containers and Orchestration (2 weeks)

#### 12.2.1 Docker
- [ ] Container concepts
- [ ] Docker images
- [ ] Dockerfile
- [ ] Building images
- [ ] Running containers
- [ ] Docker volumes
- [ ] Docker networks
- [ ] Docker Compose
- [ ] Multi-stage builds
- [ ] Best practices

#### 12.2.2 Kubernetes Basics
- [ ] Kubernetes architecture
- [ ] Pods
- [ ] Deployments
- [ ] Services
- [ ] ConfigMaps
- [ ] Secrets
- [ ] Ingress
- [ ] Horizontal Pod Autoscaler
- [ ] Kubernetes for ML workloads

#### 12.2.3 ML on Kubernetes
- [ ] Kubeflow
- [ ] KFServing
- [ ] Seldon Core
- [ ] Model scaling

---

### 12.3 CI/CD for ML (1.5 weeks)

#### 12.3.1 CI/CD Fundamentals
- [ ] Continuous Integration
- [ ] Continuous Delivery
- [ ] Continuous Deployment
- [ ] Git workflows

#### 12.3.2 ML-Specific CI/CD
- [ ] Data versioning in CI/CD
- [ ] Model versioning
- [ ] Testing ML systems
- [ ] Automated retraining
- [ ] A/B testing infrastructure
- [ ] Canary deployments
- [ ] Shadow deployments

#### 12.3.3 Tools
- [ ] GitHub Actions
- [ ] GitLab CI
- [ ] Jenkins
- [ ] Argo Workflows
- [ ] MLflow Pipelines

---

### 12.4 Model Monitoring and Versioning (1.5 weeks)

#### 12.4.1 Model Versioning
- [ ] MLflow Model Registry
- [ ] DVC (Data Version Control)
- [ ] Model cards
- [ ] Lineage tracking

#### 12.4.2 Model Monitoring
- [ ] Performance monitoring
- [ ] Data drift detection
- [ ] Concept drift detection
- [ ] Prediction distribution monitoring
- [ ] Alerting systems
- [ ] Logging strategies

#### 12.4.3 Monitoring Tools
- [ ] Evidently AI
- [ ] Arize
- [ ] WhyLabs
- [ ] Prometheus + Grafana
- [ ] Custom monitoring solutions

---

### 12.5 Cloud Platforms (2 weeks)

#### 12.5.1 AWS ML Services
- [ ] SageMaker (Training, Hosting, Processing, Pipelines, Feature Store)
- [ ] Rekognition
- [ ] Comprehend
- [ ] Lex
- [ ] Polly

#### 12.5.2 Google Cloud ML
- [ ] Vertex AI (Training, Prediction, Pipelines, Feature Store)
- [ ] AutoML
- [ ] AI Platform

#### 12.5.3 Azure ML
- [ ] Azure Machine Learning (Compute targets, Environments, Pipelines, Model registry)
- [ ] Cognitive Services

#### 12.5.4 Cloud Comparison
- [ ] Feature comparison
- [ ] Pricing considerations
- [ ] Multi-cloud strategies

---

### 12.6 Model Optimization (1.5 weeks)

#### 12.6.1 Quantization
- [ ] Post-training quantization
- [ ] Quantization-aware training
- [ ] INT8, FP16, BF16
- [ ] Dynamic quantization
- [ ] Static quantization
- [ ] Weight-Only Quantization (GPTQ, AWQ)
- [ ] 1-bit LLMs (BitNet 1.58b, T-Bit)
- [ ] Double Quantization (NF4) logic

#### 12.6.2 Pruning
- [ ] Magnitude pruning
- [ ] Structured pruning
- [ ] Unstructured pruning
- [ ] Lottery Ticket Hypothesis

#### 12.6.3 Knowledge Distillation
- [ ] Teacher-student architecture
- [ ] Soft targets
- [ ] Temperature scaling
- [ ] Distillation losses

#### 12.6.4 Neural Architecture Search (NAS)
- [ ] Search space
- [ ] Search strategy
- [ ] Performance estimation
- [ ] AutoML tools

#### 12.6.5 Compilation and Optimization
- [ ] TensorRT
- [ ] OpenVINO
- [ ] TVM
- [ ] XLA

### 12.7 Distributed Deep Learning & AI Infrastructure (1.5 weeks)

#### 12.7.1 Model & Data Parallelism
- [ ] DeepSpeed and ZeRO optimizer stages
- [ ] FSDP (Fully Sharded Data Parallel in PyTorch)
- [ ] HuggingFace Accelerate
- [ ] Megatron-LM concepts

#### 12.7.2 Custom GPU Kernels & Optimization
- [ ] CUDA Basics (Memory hierarchy, threads, blocks)
- [ ] OpenAI Triton (Writing custom GPU kernels in Python)
- [ ] FlashAttention integration

#### 12.7.3 Cluster Orchestration & Workflow
- [ ] Ray (Ray Core, Ray Serve, Ray Tune)
- [ ] Feature Stores (Feast, Hopsworks)

### 12.8 AI Security & Governance (1 week)

#### 12.8.1 Vulnerabilities & Defense
- [ ] OWASP Top 10 for LLMs
- [ ] Adversarial attacks & Prompt Injection defenses
- [ ] Data privacy & PII anonymization in pipelines

#### 12.8.2 Guardrails
- [ ] NeMo Guardrails
- [ ] Llama-Guard / Outbound filtering
- [ ] Jailbreak detection

---

## 13. Advanced Topics (8-10 weeks)

**Prerequisites:** Deep Learning, Strong mathematics

### 13.1 Reinforcement Learning (3 weeks)

#### 13.1.1 RL Fundamentals
- [ ] RL problem formulation
- [ ] Agent, environment, state, action, reward
- [ ] Markov Decision Processes (MDPs)
- [ ] Policy
- [ ] Value function
- [ ] Q-function
- [ ] Bellman equations
- [ ] Discount factor

#### 13.1.2 Tabular Methods
- [ ] Dynamic Programming (Policy iteration, Value iteration)
- [ ] Monte Carlo Methods (MC prediction, MC control)
- [ ] Temporal Difference (TD) (TD prediction, SARSA, Q-Learning)

#### 13.1.3 Function Approximation
- [ ] DQN (Deep Q-Network): Experience replay, Target networks, Double DQN, Dueling DQN, Prioritized experience replay
- [ ] Policy Gradient Methods: REINFORCE, Actor-Critic, A2C, A3C

#### 13.1.4 Advanced RL Algorithms
- [ ] PPO (Proximal Policy Optimization)
- [ ] TRPO (Trust Region Policy Optimization)
- [ ] SAC (Soft Actor-Critic)
- [ ] TD3 (Twin Delayed DDPG)
- [ ] DDPG (Deep Deterministic Policy Gradient)

#### 13.1.5 Advanced Topics
- [ ] Multi-agent RL
- [ ] Inverse RL
- [ ] Imitation learning
- [ ] Meta-RL
- [ ] Hierarchical RL
- [ ] Model-based RL

---

### 13.2 Generative AI (2 weeks)

#### 13.2.1 GANs (Deep Dive)
- [ ] Original GAN
- [ ] DCGAN
- [ ] Conditional GAN
- [ ] WGAN (Wasserstein GAN)
- [ ] WGAN-GP
- [ ] Progressive GAN
- [ ] StyleGAN series
- [ ] BigGAN
- [ ] CycleGAN
- [ ] Pix2Pix

#### 13.2.2 VAEs (Deep Dive)
- [ ] Variational inference
- [ ] Reparameterization trick
- [ ] ELBO
- [ ] Beta-VAE
- [ ] Conditional VAE
- [ ] Hierarchical VAE

#### 13.2.3 Diffusion Models (Deep Dive)
- [ ] Denoising Diffusion Probabilistic Models
- [ ] Score-based models
- [ ] Score matching
- [ ] Langevin dynamics
- [ ] Classifier-free guidance
- [ ] Latent diffusion
- [ ] Applications

#### 13.2.4 Flow Models
- [ ] Normalizing flows
- [ ] Change of variables
- [ ] RealNVP
- [ ] Glow
- [ ] Flow++

#### 13.2.5 Energy Based Models
- [ ] Energy functions
- [ ] Contrastive divergence
- [ ] Applications

---

### 13.3 Graph Neural Networks (2 weeks)

#### 13.3.1 Graph Fundamentals Review
- [ ] Graph representations
- [ ] Adjacency matrix
- [ ] Node features
- [ ] Edge features
- [ ] Graph Laplacian

#### 13.3.2 GNN Architectures
- [ ] Graph Convolutional Networks (GCN)
- [ ] GraphSAGE
- [ ] Graph Attention Networks (GAT)
- [ ] Message Passing Neural Networks (MPNN)
- [ ] GIN (Graph Isomorphism Network)

#### 13.3.3 Advanced GNNs
- [ ] Relational GCN
- [ ] Heterogeneous GNNs
- [ ] Dynamic GNNs
- [ ] Temporal GNNs

#### 13.3.4 GNN Applications
- [ ] Node classification
- [ ] Link prediction
- [ ] Graph classification
- [ ] Molecular property prediction
- [ ] Social network analysis
- [ ] Recommendation systems

---

### 13.4 Time Series Analysis (2 weeks)

#### 13.4.1 Time Series Fundamentals
- [ ] Time series components: Trend, Seasonality, Cyclical, Irregular
- [ ] Stationarity
- [ ] Autocorrelation
- [ ] Partial autocorrelation

#### 13.4.2 Classical Methods
- [ ] Moving averages
- [ ] Exponential smoothing
- [ ] ARIMA
- [ ] SARIMA
- [ ] VAR
- [ ] Prophet

#### 13.4.3 Deep Learning for Time Series
- [ ] RNNs for time series
- [ ] LSTMs for forecasting
- [ ] Temporal Convolutional Networks (TCN)
- [ ] Transformer-based models
- [ ] DeepAR
- [ ] N-BEATS
- [ ] Informer
- [ ] Autoformer

#### 13.4.4 Time Series Applications
- [ ] Forecasting
- [ ] Anomaly detection
- [ ] Classification
- [ ] Imputation

---

### 13.5 Recommendation Systems (1.5 weeks)

#### 13.5.1 Recommendation Approaches
- [ ] Content-based filtering
- [ ] Collaborative filtering (User-based, Item-based)
- [ ] Matrix factorization
- [ ] Hybrid approaches

#### 13.5.2 Deep Learning for Recommendations
- [ ] Neural Collaborative Filtering
- [ ] Autoencoders for recommendations
- [ ] Sequence-aware recommendations
- [ ] Context-aware recommendations

#### 13.5.3 Advanced Topics
- [ ] Two-tower architectures
- [ ] Wide & Deep
- [ ] DeepFM
- [ ] DIN (Deep Interest Network)
- [ ] Multi-task learning for recommendations

---

### 13.6 Federated Learning (1 week)

#### 13.6.1 FL Fundamentals
- [ ] Federated learning concept
- [ ] Privacy preservation
- [ ] Communication efficiency
- [ ] Heterogeneity challenges

#### 13.6.2 FL Algorithms
- [ ] FedAvg
- [ ] FedProx
- [ ] FedSGD
- [ ] Personalized FL

#### 13.6.3 FL Applications
- [ ] Mobile keyboard prediction
- [ ] Healthcare
- [ ] Finance

---

### 13.7 Explainable AI (XAI) (1.5 weeks)

#### 13.7.1 XAI Fundamentals
- [ ] Interpretability vs explainability
- [ ] Intrinsic vs post-hoc explanations
- [ ] Local vs global explanations
- [ ] Model-agnostic vs model-specific

#### 13.7.2 Feature Importance Methods
- [ ] Permutation importance
- [ ] SHAP (SHapley Additive exPlanations)
- [ ] LIME (Local Interpretable Model-agnostic Explanations)
- [ ] Integrated gradients
- [ ] DeepLIFT

#### 13.7.3 Visualization Methods
- [ ] Saliency maps
- [ ] Grad-CAM
- [ ] Attention visualization
- [ ] Partial dependence plots
- [ ] ICE plots

#### 13.7.4 Interpretable Models
- [ ] Decision trees
- [ ] Rule-based models
- [ ] GAMs (Generalized Additive Models)
- [ ] Explainable Boosting Machines

---

## 14. Multi-Agent Systems & Modern Architectures (6-8 weeks)

**Prerequisites:** LLMs, Deep Learning, Software Engineering

### 14.1 Agent Fundamentals (1.5 weeks)

#### 14.1.1 Agent Concepts
- [ ] What is an AI agent?
- [ ] Agent architectures
- [ ] Reactive agents
- [ ] Deliberative agents
- [ ] Hybrid agents
- [ ] Agent properties: Autonomy, Reactivity, Proactiveness, Social ability

#### 14.1.2 LLM-Based Agents
- [ ] LLM as agent core
- [ ] Planning
- [ ] Memory
- [ ] Tool use
- [ ] Reflection
- [ ] Self-improvement

#### 14.1.3 Agent Components
- [ ] Perception
- [ ] Reasoning
- [ ] Action
- [ ] Memory systems
- [ ] Learning mechanisms

---

### 14.2 LangChain Framework (2 weeks)

#### 14.2.1 LangChain Basics
- [ ] Core concepts
- [ ] Installation and setup
- [ ] LLM integration
- [ ] Prompt templates
- [ ] Output parsers

#### 14.2.2 LangChain Components
- [ ] Models (LLMs, Chat models, Embedding models)
- [ ] Prompts (Prompt templates, Few-shot prompts, Prompt selectors)
- [ ] Indexes (Document loaders, Text splitters, Vector stores, Retrievers)
- [ ] Chains (Sequential chains, Transform chains, Router chains)
- [ ] Agents (Agent types, Tools, Toolkits)
- [ ] Memory (Conversation memory, Vector store memory, Entity memory)

#### 14.2.3 Advanced LangChain
- [ ] Custom chains
- [ ] Custom agents
- [ ] Custom tools
- [ ] Callbacks
- [ ] Tracing and debugging
- [ ] LangServe
- [ ] LangSmith

---

### 14.3 LangGraph (1.5 weeks)

#### 14.3.1 Graph-Based Workflows
- [ ] State machines
- [ ] Graph representation
- [ ] Nodes and edges
- [ ] State management

#### 14.3.2 LangGraph Components
- [ ] StateGraph
- [ ] Nodes
- [ ] Edges
- [ ] Conditional edges
- [ ] State schema

#### 14.3.3 Building Multi-Agent Systems
- [ ] Agent coordination
- [ ] Message passing
- [ ] Shared state
- [ ] Orchestration patterns

---

### 14.4 Google ADK - Agent Development Kit (1 week)

#### 14.4.1 ADK Fundamentals
- [ ] Google's agent framework
- [ ] Installation
- [ ] Core concepts

#### 14.4.2 ADK Components
- [ ] Agent definition
- [ ] Tool integration
- [ ] Session management
- [ ] Deployment options

---

### 14.5 AutoGen (1.5 weeks)

#### 14.5.1 AutoGen Basics
- [ ] Microsoft AutoGen framework
- [ ] Conversable agents
- [ ] Agent types

#### 14.5.2 AutoGen Components
- [ ] AssistantAgent
- [ ] UserProxyAgent
- [ ] GroupChat
- [ ] Conversations

#### 14.5.3 Multi-Agent Patterns
- [ ] Two-agent chat
- [ ] Group chat
- [ ] Sequential workflows
- [ ] Hierarchical agents

---

### 14.6 CrewAI (1 week)

#### 14.6.1 CrewAI Fundamentals
- [ ] Role-based agents
- [ ] Task definition
- [ ] Process orchestration

#### 14.6.2 CrewAI Components
- [ ] Agent roles
- [ ] Tasks
- [ ] Crews
- [ ] Processes

---

### 14.7 Semantic Kernel (1 week)

#### 14.7.1 Semantic Kernel Basics
- [ ] Microsoft Semantic Kernel
- [ ] Skills and functions
- [ ] Planner

#### 14.7.2 Semantic Kernel Components
- [ ] Native functions
- [ ] Semantic functions
- [ ] Memory
- [ ] Orchestration

---

### 14.8 LlamaIndex (1.5 weeks)

#### 14.8.1 LlamaIndex Fundamentals
- [ ] Data framework for LLMs
- [ ] Installation
- [ ] Core concepts

#### 14.8.2 LlamaIndex Components
- [ ] Document loaders
- [ ] Node parsers
- [ ] Indexes (List index, Vector index, Keyword index, Tree index)
- [ ] Query engines
- [ ] Retrievers

#### 14.8.3 Advanced LlamaIndex
- [ ] Custom indexes
- [ ] Advanced retrieval
- [ ] Query transformations
- [ ] Integration with agents

---

### 14.9 Haystack (1 week)

#### 14.9.1 Haystack Basics
- [ ] Deepset Haystack
- [ ] Pipeline concept
- [ ] Components

#### 14.9.2 Haystack Components
- [ ] Document stores
- [ ] Retrievers
- [ ] Readers
- [ ] Generators
- [ ] Rankers

---

### 14.10 Multi-Agent Orchestration Patterns (1.5 weeks)

#### 14.10.1 Orchestration Patterns
- [ ] Centralized orchestration
- [ ] Decentralized coordination
- [ ] Hierarchical organization
- [ ] Market-based approaches
- [ ] Consensus mechanisms

#### 14.10.2 Communication Patterns
- [ ] Direct messaging
- [ ] Broadcast
- [ ] Publish-subscribe
- [ ] Blackboard systems

#### 14.10.3 Coordination Mechanisms
- [ ] Task allocation
- [ ] Resource sharing
- [ ] Conflict resolution
- [ ] Negotiation

---

### 14.11 Agent Communication Protocols (1 week)

#### 14.11.1 Communication Standards
- [ ] FIPA ACL
- [ ] KQML
- [ ] Custom protocols

#### 14.11.2 Message Formats
- [ ] JSON-based
- [ ] Protocol buffers
- [ ] Custom schemas

---

### 14.12 Tool Use and Function Calling (1.5 weeks)

#### 14.12.1 Tool Use Fundamentals
- [ ] Tool definition
- [ ] Tool discovery
- [ ] Tool selection
- [ ] Tool execution

#### 14.12.2 Function Calling
- [ ] OpenAI function calling
- [ ] Tool calling APIs
- [ ] Schema definition
- [ ] Error handling

#### 14.12.3 Advanced Tool Use
- [ ] Multi-tool workflows
- [ ] Tool composition
- [ ] Dynamic tool creation
- [ ] Tool learning

---

# PHASE 6: PROFESSIONAL DEVELOPMENT (Ongoing)

## 15. Tools & Technologies (Ongoing)

### 15.1 Git & GitHub

#### 15.1.1 Git Fundamentals
- [ ] Version control concepts
- [ ] Git installation
- [ ] Repository initialization
- [ ] Staging and committing
- [ ] Viewing history
- [ ] Branching
- [ ] Merging
- [ ] Rebasing
- [ ] Remote repositories
- [ ] Push and pull
- [ ] Fetch
- [ ] Cloning

#### 15.1.2 Advanced Git
- [ ] Cherry-picking
- [ ] Stashing
- [ ] Tagging
- [ ] Submodules
- [ ] Git hooks
- [ ] Interactive rebase
- [ ] Bisect
- [ ] Reflog

#### 15.1.3 GitHub
- [ ] Pull requests
- [ ] Code review
- [ ] Issues
- [ ] Projects
- [ ] Actions
- [ ] Pages
- [ ] Gists
- [ ] GitHub CLI

---

### 15.2 Linux Basics

#### 15.2.1 Command Line
- [ ] Navigation: cd, ls, pwd
- [ ] File operations: cp, mv, rm, mkdir
- [ ] Viewing files: cat, less, head, tail
- [ ] Text processing: grep, sed, awk
- [ ] Permissions: chmod, chown
- [ ] Process management: ps, top, kill
- [ ] Networking: ping, curl, wget, ssh
- [ ] Package management: apt, yum, pip

#### 15.2.2 Shell Scripting
- [ ] Bash basics
- [ ] Variables
- [ ] Conditionals
- [ ] Loops
- [ ] Functions
- [ ] Script automation

---

### 15.3 IDEs and Editors

#### 15.3.1 VS Code
- [ ] Installation and setup
- [ ] Extensions
- [ ] Debugging
- [ ] Integrated terminal
- [ ] Git integration
- [ ] Python development
- [ ] Jupyter notebooks
- [ ] Remote development

#### 15.3.2 PyCharm
- [ ] Project setup
- [ ] Debugging
- [ ] Refactoring
- [ ] Testing
- [ ] Database tools

#### 15.3.3 JupyterLab
- [ ] Advanced features
- [ ] Extensions
- [ ] Customization

---

### 15.4 Experiment Tracking

#### 15.4.1 Weights & Biases
- [ ] Installation
- [ ] Logging metrics
- [ ] Logging artifacts
- [ ] Hyperparameter sweeps
- [ ] Reports
- [ ] Team collaboration

#### 15.4.2 TensorBoard
- [ ] Installation
- [ ] Scalars
- [ ] Graphs
- [ ] Images
- [ ] Histograms
- [ ] Projector

#### 15.4.3 MLflow
- [ ] Tracking
- [ ] Projects
- [ ] Models
- [ ] Registry

#### 15.4.4 Other Tools
- [ ] Neptune
- [ ] Comet.ml
- [ ] Aim

---

### 15.5 Model Registries

#### 15.5.1 MLflow Model Registry
- [ ] Model registration
- [ ] Versioning
- [ ] Stage transitions
- [ ] Annotations

#### 15.5.2 Other Registries
- [ ] AWS SageMaker Registry
- [ ] Azure ML Registry
- [ ] Hugging Face Model Hub

---

### 15.6 Vector Databases

#### 15.6.1 Pinecone
- [ ] Index creation
- [ ] Upserting vectors
- [ ] Querying
- [ ] Metadata filtering

#### 15.6.2 Milvus
- [ ] Collections
- [ ] Indexes
- [ ] Search
- [ ] Filtering

#### 15.6.3 Chroma
- [ ] Collections
- [ ] Embedding functions
- [ ] Querying

#### 15.6.4 Weaviate
- [ ] Schema
- [ ] Objects
- [ ] GraphQL queries
- [ ] Modules

#### 15.6.5 Qdrant
- [ ] Collections
- [ ] Points
- [ ] Search
- [ ] Filtering

---

### 15.7 Message Queues

#### 15.7.1 Kafka
- [ ] Concepts: topics, partitions, brokers
- [ ] Producers
- [ ] Consumers
- [ ] Consumer groups
- [ ] Kafka Streams

#### 15.7.2 RabbitMQ
- [ ] Exchanges
- [ ] Queues
- [ ] Bindings
- [ ] Consumers
- [ ] Message patterns

#### 15.7.3 Redis & Celery (Task Queues)
- [ ] Celery architecture (Brokers & Backends)
- [ ] Asynchronous task processing
- [ ] Scheduled tasks (Celery Beat)
- [ ] Using Redis as a message broker
- [ ] Using Redis for caching model results

---

## 16. Projects & Portfolio (Ongoing)

### 16.1 Suggested Projects by Level

#### 16.1.1 Beginner Projects
- [ ] Titanic survival prediction
- [ ] House price prediction
- [ ] Iris classification
- [ ] MNIST digit classification
- [ ] Spam email classifier
- [ ] Sentiment analysis on movie reviews
- [ ] Customer segmentation
- [ ] Handwritten digit recognition

#### 16.1.2 Intermediate Projects
- [ ] Stock price prediction
- [ ] Chatbot implementation
- [ ] Image captioning
- [ ] Object detection system
- [ ] Recommendation system
- [ ] Fraud detection
- [ ] Disease prediction
- [ ] Text summarization
- [ ] Face recognition system
- [ ] Music genre classification

#### 16.1.3 Advanced Projects
- [ ] End-to-end ML pipeline
- [ ] Real-time object detection
- [ ] Conversational AI assistant
- [ ] Multi-modal model
- [ ] Custom GPT application
- [ ] RAG system implementation
- [ ] Fine-tuned LLM
- [ ] Production deployment with monitoring
- [ ] Distributed training implementation
- [ ] Custom neural architecture

---

### 16.2 Kaggle Competitions

#### 16.2.1 Getting Started
- [ ] Creating account
- [ ] Understanding competitions
- [ ] Submission format
- [ ] Leaderboards

#### 16.2.2 Competition Types
- [ ] Getting Started competitions
- [ ] Playground competitions
- [ ] Featured competitions
- [ ] Research competitions

#### 16.2.3 Competition Strategy
- [ ] EDA
- [ ] Baseline model
- [ ] Feature engineering
- [ ] Model selection
- [ ] Ensembling
- [ ] Cross-validation
- [ ] Time management

---

### 16.3 Open Source Contributions

#### 16.3.1 Finding Projects
- [ ] GitHub explore
- [ ] Good first issues
- [ ] ML libraries to consider: scikit-learn, PyTorch, TensorFlow, Hugging Face transformers, LangChain, LlamaIndex

#### 16.3.2 Contribution Types
- [ ] Bug fixes
- [ ] Documentation
- [ ] Feature implementation
- [ ] Tests
- [ ] Examples

---

### 16.4 Research Paper Reading

#### 16.4.1 Paper Sources
- [ ] arXiv
- [ ] NeurIPS
- [ ] ICML
- [ ] ICLR
- [ ] CVPR
- [ ] ACL
- [ ] EMNLP

#### 16.4.2 Reading Strategy
- [ ] Title and abstract
- [ ] Introduction
- [ ] Figures and tables
- [ ] Methodology
- [ ] Experiments
- [ ] Conclusion
- [ ] Related work

#### 16.4.3 Paper Implementation
- [ ] Understanding code
- [ ] Reproducing results
- [ ] Experimentation

---

# Learning Path Summary

## Phase Timeline

| Phase | Duration | Focus Areas |
|-------|----------|-------------|
| Phase 1 | Months 1-4 | Mathematics, Python, Data Science Libraries |
| Phase 2 | Months 5-7 | Data Engineering, DSA, ML Fundamentals |
| Phase 3 | Months 8-12 | Classical ML, Deep Learning, Frameworks |
| Phase 4 | Months 13-18 | NLP or Computer Vision Specialization |
| Phase 5 | Months 19-24 | MLOps, Advanced Topics, Multi-Agent Systems |
| Phase 6 | Ongoing | Tools, Projects, Portfolio Building |

## Key Dependencies

```
Mathematics → ML Algorithms → Deep Learning → Specialization
     ↓              ↓              ↓              ↓
Python → Data Science → Classical ML → Advanced Topics
     ↓              ↓              ↓              ↓
DSA → Data Engineering → MLOps → Production Systems
```

## Success Metrics

- [ ] Complete all foundational topics
- [ ] Build 10+ substantial projects
- [ ] Contribute to 2+ open source projects
- [ ] Read and implement 20+ research papers
- [ ] Participate in 5+ Kaggle competitions
- [ ] Deploy 3+ production-ready systems
- [ ] Master 2+ specialization areas

---

**This roadmap provides a comprehensive path from beginner to expert in AI/ML. The journey requires dedication, consistent practice, and hands-on implementation. Remember that depth of understanding is more important than speed of completion.**
