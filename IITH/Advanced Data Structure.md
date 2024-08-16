Links: [IIT Hyderabad](IIT%20Hyderabad.md), [Data Structures & Algorithms](../GATE%20Prep/Data%20Structures%20&%20Algorithms.md)

# ADS - 01

## How Algorithms Should work: (Students Opinion)
- Correctness -> It should Provide Correct Data
- Good Space & Time Complexity
- Finite Steps / Operations
- Unambiguous -> Clear / Precise
- Should not be Hardware or Software Dependent
- Must Have output
## Deterministic & Randomize Algorithms

### **Deterministic Algorithm:**
- Always gives the same result for the same input.
- Steps are fixed and predictable.

  **Examples:**
  - **Bubble Sort:** Always sorts the same list in the same way.
  - **Dijkstra's Algorithm:** Always finds the shortest path in a graph with fixed weights.

### **Randomized Algorithm:**
- Uses randomness as part of its logic.
- Can give different results or take different steps on the same input.

  **Examples:**
  - **Randomized Quick Sort:** Uses a randomly chosen pivot, which can lead to different performance on the same data.
  - **Monte Carlo Method:** Uses random sampling to solve problems that might be deterministic in nature.

### LLM Example:
Large Language Models (LLMs) like GPT-4 can exhibit both deterministic and randomized behavior, depending on how they are used:

- **Deterministic Mode:**
  - When configured with a fixed random seed, the model produces the same output for the same input every time.
  - This can be useful for debugging and testing.

- **Randomized Mode:**
  - When no fixed seed is set, the model's outputs can vary for the same input due to the inherent randomness in the sampling process.
  - This allows for more diverse and creative responses.

In practice, **LLMs often operate in a randomized mode** to provide varied and contextually rich responses.

## Extra

- Sorting Algo
- Lower / Upper Bound -> 
	- Lower Bound Lowest Time needed (Best Time Complexity) $$ \Omega = Omega $$
	- Upper Bound Max Time Needed (Worst Time Complexity)
	  $$ O = Big O $$
	- Average Time Complexity
	  $$\theta = Theta(Average \space Time \space Complexity)$$

# ADS - 02
## Good Algorithms:
- We should take multiple comparisons to mark it as good performance algorithm.
- We have to check the performance based on High value of n (Input value), so that we can view the graph on how well it actually performs & how much time it needed. e.g. n>10^4 , 10^1000 ...
- while checking for time taken by algorithm we generally neglect constant values, because if value of n increases exponentially suppose 10^5 or more than constant value time does not matter that much.
	- **So we remove lower order term & constant values & multiples and just take the higher order of n for checking.**
		- f(n)=5n^2+3n+7
		- **Higher-order term:** 5n^2 (5 is Multiple, so we are only taking n^2 in account)
		- **Lower-order term:** 3n
		- **Constant:** 7

## Notations: Omega, Theta, Big O

Yes, each of these time complexity notations is defined mathematically with specific equations.
> 	n_0 means that after certain value of n the equation will be always true, that value is point of n(input) which decides the time complexity of function (Graphs are also available.)

### 1. **Big-O Notation \( O(g(n)) \)**:
   - **Definition**: A function \( f(n) \) is \( O(g(n)) \) if there exist positive constants \( c \) and \( n_0 \) such that:
   - 
 $$  [
   f(n) \leq c \cdot g(n) \quad \text{for all } n \geq n_0
   ]$$
   - **Interpretation**: \( f(n) \) grows at most as fast as \( g(n) \) for sufficiently large \( n \).

### 2. **Theta Notation \( \Theta(g(n)) \)**:
   - **Definition**: A function \( f(n) \) is \( \Theta(g(n)) \) if there exist positive constants \( c_1 \), \( c_2 \), and \( n_0 \) such that:
  $$ [
   c_1 \cdot g(n) \leq f(n) \leq c_2 \cdot g(n) \quad \text{for all } n \geq n_0
   ]$$
   - **Interpretation**: \( f(n) \) grows exactly as fast as \( g(n) \) for sufficiently large \( n \).

### 3. **Omega Notation \( \Omega(g(n)) \)**:
   - **Definition**: A function \( f(n) \) is \( \Omega(g(n)) \) if there exist positive constants \( c \) and \( n_0 \) such that:
   $$[
   f(n) \geq c \cdot g(n) \quad \text{for all } n \geq n_0
   ]$$
   - **Interpretation**: \( f(n) \) grows at least as fast as \( g(n) \) for sufficiently large \( n \).

![Time Complexity Notation](../Archive/Attachment/ADS%20Time%20Complexity%20Notation.png)

### **Summary of the Equations:**

- **Big-O $$( O(g(n)) ): ( f(n) \leq c \cdot g(n) )$$
- **Theta $$( \Theta(g(n)) ): ( c_1 \cdot g(n) \leq f(n) \leq c_2 \cdot g(n) )$$
- **Omega $$( \Omega(g(n)) ): ( f(n) \geq c \cdot g(n) )$$
![My Notes on Time Complexity](../Archive/Attachment/Time%20Complexity%20Notes.png)

These mathematical definitions formalize the relationships between \( f(n) \) and \( g(n) \) in terms of growth rates, where \( f(n) \) represents the actual time complexity of an algorithm and \( g(n) \) is the comparison function. Lower the value means lower the time Complexity of Equation.
> We are comparing the them & based on notation we know that after certain n value equation will be always true for future scenario.

# ADS - 03

- Algo A: O (n^2) &  Algo B: O (n^3)
- In Given case we will think that Algo A is better than Algo B, because of Worst case time complexity
- But Theta (Avg Case) Time Complexity of those algorithms are
$$Algo A = \theta (n^2)$$$$Algo B = \theta(n)$$
- Then that means that the Algo B is much better than Algo A.
### Quick Overview

1. **Alpha Probability**: 
   - Alpha probability typically refers to the likelihood (or probability) that a particular element is in a given list. This concept is often used in searching algorithms and statistical analysis.

2. **AVL Tree**:
   - **Definition**: A self-balancing binary search tree where the difference in heights between the left and right sub trees (known as the balance factor) is at most 1 for every node.
   - **Operations**: 
     - Insertion, deletion, and search operations take \(O(\log n)\) time because the tree maintains balance after each operation.
   - **Use**: AVL trees are used when you need efficient lookups, insertions, and deletions.

3. **Expected Value \(E(X)\)**:
   - **Formula**: $$E(X) = \sum_{i=1}^{n} P(X_i) \times X_i$$
   - **Explanation**: The expected value of a random variable \(X\) is the sum of each possible value of \(X\) weighted by its probability. For example, if a dice has faces 1 to 6, and each face has an equal probability, \(E(X)\) is the average of all possible outcomes.

4. **Randomized Algorithm/Value**:
   - **Definition**: An algorithm that uses randomness as part of its logic, often to achieve better average-case performance or to simplify implementation.
   - **Example**: QuickSort's pivot selection can be randomized to avoid worst-case scenarios on sorted input.

5. **Insertion Sort Average Time Complexity**:
   - **Time Complexity**: \(O(n^2)\)
   - **Explanation**: Insertion Sort compares each element with the previous elements and places it in the correct position. On average, it requires \(O(n^2)\) comparisons and swaps, where \(n\) is the number of elements in the list.

# ADS - 04

### Advanced Data Structures & Algorithms Concepts

---

#### 1. **Big O and Small o Notation**

- **Big O Notation**:
  - **Definition**: Big O notation, \( O(f(n)) \), describes the upper bound of an algorithm's runtime. It provides the worst-case scenario, showing how the runtime grows as the input size \( n \) increases.
  - **Example**: If an algorithm's time complexity is \( O(n^2) \), its runtime grows quadratically with the input size.

- **Small o Notation**:
  - **Definition**: Small o notation, \( o(f(n)) \), describes a stricter upper bound than Big O. It indicates that the algorithm's runtime grows slower than \( f(n) \) as the input size increases.
  - **Example**: If \( T(n) = o(n^2) \), the runtime grows slower than \( n^2 \), but it doesn’t reach \( n^2 \) as \( n \) becomes large.

---

#### 2. **Average Case Analysis of Insertion Sort**

- **Insertion Sort Overview**:
  - Insertion Sort is a simple sorting algorithm that builds the final sorted array one element at a time.
  - It works by repeatedly taking the next element and inserting it into the correct position in the already sorted part of the array.

- **Average Case Analysis**:
  - The average case occurs when the elements are in a random order.
  - **Key Insight**: On average, half of the elements in the sorted portion of the array need to be compared and shifted for each insertion.
  - **Average Case Time Complexity**: 
    \[
    T_{avg}(n) = O(n^2)
    \]
  - **Steps**:
    1. For each element, the inner loop may have to shift, on average, \( n/2 \) elements.
    2. Summing over all elements gives an average complexity of \( O(n^2) \).

---

#### 3. **Randomized Quick Sort**

- **Quick Sort Overview**:
  - Quick Sort is a divide-and-conquer sorting algorithm. It picks a "pivot" element, partitions the array around the pivot, and then recursively sorts the subarrays.

- **Randomized Quick Sort**:
  - **Randomization**: Instead of always choosing a fixed pivot (e.g., the first or last element), Randomized Quick Sort picks a pivot at random from the current subarray.
  - **Advantage**: Randomizing the pivot selection helps avoid the worst-case scenario (which is \( O(n^2) \) for a non-randomized Quick Sort when the array is already sorted or nearly sorted).
  - **Expected Time Complexity**:
    $$[
    T_{avg}(n) = O(n \log n)
    ]$$
  - **Reasoning**: The expected depth of the recursion tree is \( \log n \), and each level of the tree requires \( O(n) \) work for partitioning.

---

#### 4. **Probability & Permutation**

- **Probability**:
  - **Basic Definition**: Probability measures the likelihood of a specific event occurring, defined as:
    $$[
    P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}
    ]$$
  - **Example**: The probability of rolling a 4 on a 6-sided die is $( \frac{1}{6} ).$

- **Permutation**:
  - **Definition**: A permutation is an arrangement of all or part of a set of objects. The number of permutations of \( n \) distinct objects taken \( r \) at a time is given by:
    $$[
    P(n, r) = \frac{n!}{(n-r)!}
    ]$$
  - **Example**: The number of ways to arrange 3 letters out of the set \{A, B, C, D\} is \( P(4, 3) = 24 \).

---

#### 5. **Binary Search Tree (BST)**

- **Definition**:
  - A Binary Search Tree is a binary tree where each node has at most two children.
  - For any node \( N \):
    - All elements in the left subtree of \( N \) are less than \( N \).
    - All elements in the right subtree of \( N \) are greater than \( N \).

![Binary Search Tree](../Archive/Attachment/Binary%20Search%20tree.png)
- **Key Operations**:
  - **Insertion**: Start from the root and insert the new node in the correct position such that the BST property is maintained.
  - **Search**: Start from the root and recursively move left or right, depending on whether the value is smaller or larger than the current node.
  - **Deletion**: There are three cases to handle:
    1. Node to be deleted is a leaf (has no children).
    2. Node to be deleted has one child.
    3. Node to be deleted has two children (replace with the in-order successor or predecessor).

- **Time Complexity**:
  - **Best Case** (balanced tree): \( O(\log n) \)
  - **Worst Case** (unbalanced tree, e.g., a linked list): \( O(n) \)

# ADS - 05

## Randomized Quick Sort:
### Info

- **Randomized Quick Sort Expected Time Complexity**: $$O(nlog⁡n)$$
- **Probability of Selecting Any Specific Number as Pivot**: $$\frac{1}{n}​$$
### A[i] & A[j] not being compared

- If we take into account that List is already sorted & then we select A[i] & A[j] element in **randomized Quick sort algorithm** they will only not get compared. If the pivot point is in between this no.
$$A=\left[ 1, \frac{3}{i} ,5,7, \frac{9}{j} ,10 \right]$$
- Here A[i]=3 & A[j]=9, they will not get compared if Pivot is = [5 , 7].
- **Correctness**: The formula $$\frac{2}{j - i + 1}$$​ provides the probability that two specific elements A[i] and A[j] are compared during a single partitioning step. It is accurate for calculating the probability of comparison in that specific context.
- **Asymptotic Complexity**: The total expected number of comparisons for all pairs in Randomized Quick Sort is $$Θ(nlog⁡n)$$, which reflects the overall complexity of the algorithm.

### Min-Cut Problem:
The **min-cut problem** in graph theory involves finding the smallest set of edges that, when removed, separates the graph into two disconnected components (subgraphs). 
> We are going to discuss it in next lecture

# ADS - 06

## Min-Cut Problem:
Suppose we have graph (Node and edges), we want to find the minimum cuts (edges remove) so that that will divide into two graphs.
- A **graph** is a mathematical structure consisting of **nodes** (vertices) and **edges** (connections between nodes).
- The Graph is a pictorial representation or a diagram that represents data or values in an organized manner.

![Min Cut Problem](../Archive/Attachment/Min%20Cut%20Problem.png)
### **To Solve it :**
- We will use **Karger's Algorithm**, a randomized algorithm used to solve the **min-cut problem** in an undirected, unweighted graph. 

### Steps of Karger's Algorithm:

1. **Pick a Random Edge**: 
   - Uniformly pick an edge at random from the graph.

2. **Contract the Edge**:
   - Merge the two vertices connected by the selected edge into a single vertex.
   - This contraction may create self-loops (edges that connect the vertex to itself), which should be removed.
   - Any other edges between the merged vertex and other vertices are retained.

3. **Repeat the Process**:
   - Continue picking random edges and contracting them until only two vertices remain in the graph.

4. **Count the Edges**:
   - The number of edges between the last two remaining vertices is the size of a cut that separates the graph into two parts.

![Kargers Algo Min-Cut](../Archive/Attachment/Kargers%20Algo.png)

### Result:
- The edges left between the final two vertices represent a cut in the original graph. But that answer is not exact answer true minimum cut can be different.
- The algorithm is repeated multiple times to increase the probability of finding the true minimum cut, as it's a randomized algorithm.

### Claim:

1. **Intermediate Graphs & Min-Cut:**
   - In Karger's algorithm, every intermediate graph has a minimum cut size that is **at least** the size of the original graph's minimum cut.

2. **Number of Edges (|G| = k):**
   - **Conditionally True:** The number of edges in the graph ( G ) is at least $$( \frac{nk}{2} )$$ (where ( n ) is the number of nodes and ( k ) is the min-cut value). This relationship depends on the graph's structure and may not apply universally.
3. Probability that an edge 'X' (which is required for minimum cut) is not picked in first time is 1 - (2/n).

# ADS - 07

## Algorithm Correctness

#### 1. **f(x) = g(x) Check:**
   - **Goal:** Determine if two functions \( f(x) \) and \( g(x) \) are identical.
   - **Approach:** 
     - **Randomized Testing:** 
       - Choose a random number \( r \) in the range \([0, 1000d]\).
       - Evaluate the polynomial \( P(r) = f(r) - g(r) \).
       - If \( P(r) = 0 \), declare \( f(x) = g(x) \) (identical).
       - If \( P(r) \neq 0 \), declare \( f(x) \neq g(x) \) (not identical).
     - **Correctness:** The correctness of this randomized approach depends on how likely it is that the functions are identical or not. If the algorithm says "identical," it is checking whether \( r \) is a root of \( P(x) \). The probability of falsely declaring functions identical is minimized by testing multiple roots.

#### 2. **Polynomial Identity Testing (PIT):**
- This is the process where the algorithm checks if \( f(x) = g(x) \) by evaluating at random points. PIT is used in algorithms to probabilistically determine if two polynomials are identical without explicitly expanding them.

#### 3. **Correctness of Algorithms:**
   - **Verifier for Algorithm Output:**
     - To check if the output of an algorithm is correct. For instance, in sorting, a verifier would check that the output list is sorted and contains all the elements of the original list.

### Key Takeaways:

- **Randomized Algorithms:** These use randomness to improve performance or correctness with high probability, though they might require multiple runs to reduce error chances.
  
- **Correctness Proof:** Algorithms like insertion sort have clear, deterministic proofs of correctness, ensuring that they always produce the correct result.

- **Verification:** Some algorithms (especially randomized ones) may require additional steps to verify that their output is correct.

### Recursive Programs

**Recursive Programs** are those that solve a problem by solving smaller instances of the same problem. In recursion, a function calls itself to solve these smaller instances, usually with a base case to terminate the recursion and avoid infinite loops.

#### Key Components of Recursion:
1. **Base Case**: The condition under which the recursion stops. It's crucial to prevent infinite recursion. For example, in calculating factorial, the base case is \( n = 0 \) or \( n = 1 \), where the function returns 1.

2. **Recursive Case**: The part of the function where it calls itself with a modified argument to reduce the problem size. 

#### Example: Factorial Function

function to calculate the factorial of a number \( n \):

```python
def factorial(n):
    if n == 0:  # Base case
        return 1
    else:       # Recursive case
        return n * factorial(n - 1)
```

### Invariants

**Invariants** are properties or conditions that remain true throughout the execution of an algorithm or during a particular part of a program. They are used to reason about the correctness of algorithms, especially in loops and recursive functions.  We can also call it as pseudo code which will be same as logic irrespective of language.

#### Types of Invariants:

1. **Loop Invariant**: A condition that holds true before and after each iteration of a loop. It's used to prove the correctness of iterative algorithms.

2. **Recursive Invariant**: A condition that remains true across recursive calls. It helps in proving the correctness of recursive algorithms.

3. **Data Invariants**: Data invariants are properties that must always hold true for the data structures used in a program.

### Constructs

**Constructs** refer to the fundamental building blocks or elements used in programming to create algorithms and programs. These include:

1. **Conditional Constructs**: Used for decision-making. Examples are `if`, `else`, and `switch` statements.

2. **Loop Constructs**: Used for repeating tasks. Examples include `for`, `while`, and `do-while` loops.

3. **Function Constructs**: Define reusable blocks of code. Examples include function declarations and definitions.

4. **Data Structures**: Constructs for storing and organizing data. Examples include arrays, linked lists, stacks, and queues.

5. **Object-Oriented Constructs**: Used in object-oriented programming (OOP) to model real-world entities. Examples include classes, objects, inheritance, and polymorphism.

# ADS - 08

## Algorithms Overview

#### 1. **Deterministic Algorithms**
- **Definition**: Algorithms that produce the same output for a given input every time they are executed.
- **Characteristics**:
  - Predictable results.
  - Often straightforward but can be less efficient in certain cases.

#### 2. **Randomized Algorithms**
- **Definition**: Algorithms that use randomization to make decisions or solve problems, and may produce different outputs on different runs with the same input.
- **Characteristics**:
  - **Probability of Correctness**: They have a certain probability of providing a correct answer. For instance, if the algorithm has a probability \( \alpha_1 \) of giving the correct answer and \( \alpha_2 \) of giving the wrong answer, \( \alpha_1 + \alpha_2 = 1 \).
  - **Optimization**: By optimizing the randomized algorithm (e.g., using more samples or better random choices), it is often possible to increase the probability of obtaining a correct result and potentially achieve better performance than some deterministic algorithms.
  - **Efficiency**: In practice, randomized algorithms can be more efficient than deterministic ones, especially for complex problems where deterministic approaches are too slow or infeasible.

#### 3. **Quantum Algorithms**
- **Definition**: Algorithms that leverage principles of quantum mechanics to solve problems, using quantum bits (qubits) and quantum operations.
- **Characteristics**:
  - **Quantum Superposition**: Quantum algorithms can process a superposition of states, allowing them to explore many possible solutions simultaneously.
  - **Interference**: Quantum algorithms use interference to amplify the probability of correct answers and cancel out incorrect ones.
  - **Probability of Correctness**: Quantum algorithms can be designed to reduce or eliminate the probability of incorrect answers by strategically using quantum operations. For example, quantum algorithms can cancel out the probability of incorrect answers (\( \alpha_2 \)) and ensure that the output is correct with high probability (\( \alpha_1 \)).
  - **Example**: Shor's algorithm for factoring large numbers and Grover's algorithm for searching an unsorted database are examples where quantum algorithms offer significant speedups over classical algorithms.
The discussion in your lecture about checking if a number is prime and the use of the square root of \( N \) in algorithms is quite relevant in number theory and algorithm design. Here's a structured note on this topic:

---

## Prime Number Testing and the Square Root Method

#### **1. Prime Number Testing**

**Definition**: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

**Goal**: To determine if a given number \( N \) is prime.

#### **2. Basic Approach**

- **Direct Method**: Check if \( N \) is divisible by any number from 2 up to \( N-1 \). If it is, \( N \) is not prime. This approach is inefficient for large numbers.

#### **3. Optimized Approach Using Square Root**

**Concept**: A more efficient method involves checking divisibility only up to \( $$\sqrt{N} $$\). The rationale is based on the following observation:

- **Observation**: If \( N \) is divisible by some number \( d \), then \( N = d \times k \), where \( k \) is also a divisor. If both \( d \) and \( k \) were greater than \( \sqrt{N} \), their product \( d \times k \) would be greater than \( N \). Hence, at least one of these divisors must be less than or equal to \( \sqrt{N} \).

**Pseudocode**:

```python
import math

def is_prime(N):
    if N <= 1:
        return False
    if N <= 3:
        return True
    if N % 2 == 0 or N % 3 == 0:
        return False
    i = 5 # because We already checked till 4
    while i * i <= N:
        if N % i == 0 or N % (i + 2) == 0:
            return False
        i += 6 # bec already checked //2 //3 and +2 so, +6 here
    return True
```

**Algorithm**:
1. **Handle Small Cases**: Check if \( N \) is less than 2. If so, it is not prime.
2. **Check Divisibility**:
   - For each integer \( i \) from 2 up to \( \sqrt{N} \):
     - If \( N \) is divisible by \( i \), then \( N \) is not a prime.
   - If no divisors are found in this range, \( N \) is a prime.
