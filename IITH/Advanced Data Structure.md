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

### 1. **Big-O Notation ( O(g(n)) )**:
   - **Definition**: A function ( f(n) ) is ( O(g(n)) ) if there exist positive constants ( c ) and ( n_0 ) such that:
   - 
 $$  [
   f(n) \leq c \cdot g(n) \quad \text{for all } n \geq n_0
   ]$$
   - **Interpretation**: ( f(n) ) grows at most as fast as ( g(n) ) for sufficiently large ( n ).

### 2. **Theta Notation ( \Theta(g(n)) )**:
   - **Definition**: A function ( f(n) ) is ( \Theta(g(n)) ) if there exist positive constants ( c_1 ), ( c_2 ), and ( n_0 ) such that:
  $$ [
   c_1 \cdot g(n) \leq f(n) \leq c_2 \cdot g(n) \quad \text{for all } n \geq n_0
   ]$$
   - **Interpretation**: ( f(n) ) grows exactly as fast as ( g(n) ) for sufficiently large ( n ).

### 3. **Omega Notation ( \Omega(g(n)) )**:
   - **Definition**: A function ( f(n) ) is ( \Omega(g(n)) ) if there exist positive constants ( c ) and ( n_0 ) such that:
   $$[
   f(n) \geq c \cdot g(n) \quad \text{for all } n \geq n_0
   ]$$
   - **Interpretation**: ( f(n) ) grows at least as fast as ( g(n) ) for sufficiently large ( n ).

![Time Complexity Notation](../Archive/Attachment/ADS%20Time%20Complexity%20Notation.png)

### **Summary of the Equations:**

- **Big-O $$( O(g(n)) ): ( f(n) \leq c \cdot g(n) )$$
- **Theta $$( \Theta(g(n)) ): ( c_1 \cdot g(n) \leq f(n) \leq c_2 \cdot g(n) )$$
- **Omega $$( \Omega(g(n)) ): ( f(n) \geq c \cdot g(n) )$$
![My Notes on Time Complexity](../Archive/Attachment/Time%20Complexity%20Notes.png)

These mathematical definitions formalize the relationships between \( f(n) ) and \( g(n) ) in terms of growth rates, where \( f(n) ) represents the actual time complexity of an algorithm and \( g(n) ) is the comparison function. Lower the value means lower the time Complexity of Equation.
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
     - Insertion, deletion, and search operations take \(O(\log n)) time because the tree maintains balance after each operation.
   - **Use**: AVL trees are used when you need efficient lookups, insertions, and deletions.

3. **Expected Value \(E(X))**:
   - **Formula**: $$E(X) = \sum_{i=1}^{n} P(X_i) \times X_i$$
   - **Explanation**: The expected value of a random variable \(X) is the sum of each possible value of \(X) weighted by its probability. For example, if a dice has faces 1 to 6, and each face has an equal probability, \(E(X)) is the average of all possible outcomes.

4. **Randomized Algorithm/Value**:
   - **Definition**: An algorithm that uses randomness as part of its logic, often to achieve better average-case performance or to simplify implementation.
   - **Example**: QuickSort's pivot selection can be randomized to avoid worst-case scenarios on sorted input.

5. **Insertion Sort Average Time Complexity**:
   - **Time Complexity**: \(O(n^2))
   - **Explanation**: Insertion Sort compares each element with the previous elements and places it in the correct position. On average, it requires \(O(n^2)) comparisons and swaps, where \(n) is the number of elements in the list.

# ADS - 04

### Advanced Data Structures & Algorithms Concepts

---

#### 1. **Big O and Small o Notation**

- **Big O Notation**:
  - **Definition**: Big O notation, \( O(f(n)) ), describes the upper bound of an algorithm's runtime. It provides the worst-case scenario, showing how the runtime grows as the input size \( n ) increases.
  - **Example**: If an algorithm's time complexity is \( O(n^2) ), its runtime grows quadratically with the input size.

- **Small o Notation**:
  - **Definition**: Small o notation, \( o(f(n)) ), describes a stricter upper bound than Big O. It indicates that the algorithm's runtime grows slower than \( f(n) ) as the input size increases.
  - **Example**: If \( T(n) = o(n^2) ), the runtime grows slower than \( n^2 ), but it doesn’t reach \( n^2 ) as \( n ) becomes large.

---

#### 2. **Average Case Analysis of Insertion Sort**

- **Insertion Sort Overview**:
  - Insertion Sort is a simple sorting algorithm that builds the final sorted array one element at a time.
  - It works by repeatedly taking the next element and inserting it into the correct position in the already sorted part of the array.

- **Average Case Analysis**:
  - The average case occurs when the elements are in a random order.
  - **Key Insight**: On average, half of the elements in the sorted portion of the array need to be compared and shifted for each insertion.
  - **Average Case Time Complexity**: 
    [
    T_{avg}(n) = O(n^2)
    ]
  - **Steps**:
    1. For each element, the inner loop may have to shift, on average, \( n/2 ) elements.
    2. Summing over all elements gives an average complexity of \( O(n^2) ).

---

#### 3. **Randomized Quick Sort**

- **Quick Sort Overview**:
  - Quick Sort is a divide-and-conquer sorting algorithm. It picks a "pivot" element, partitions the array around the pivot, and then recursively sorts the subarrays.

- **Randomized Quick Sort**:
  - **Randomization**: Instead of always choosing a fixed pivot (e.g., the first or last element), Randomized Quick Sort picks a pivot at random from the current subarray.
  - **Advantage**: Randomizing the pivot selection helps avoid the worst-case scenario (which is \( O(n^2) ) for a non-randomized Quick Sort when the array is already sorted or nearly sorted).
  - **Expected Time Complexity**:
    $$[
    T_{avg}(n) = O(n \log n)
    ]$$
  - **Reasoning**: The expected depth of the recursion tree is \( \log n ), and each level of the tree requires \( O(n) ) work for partitioning.

---

#### 4. **Probability & Permutation**

- **Probability**:
  - **Basic Definition**: Probability measures the likelihood of a specific event occurring, defined as:
    $$[
    P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}
    ]$$
  - **Example**: The probability of rolling a 4 on a 6-sided die is $( \frac{1}{6} ).$

- **Permutation**:
  - **Definition**: A permutation is an arrangement of all or part of a set of objects. The number of permutations of \( n ) distinct objects taken \( r ) at a time is given by:
    $$[
    P(n, r) = \frac{n!}{(n-r)!}
    ]$$
  - **Example**: The number of ways to arrange 3 letters out of the set \{A, B, C, D\} is \( P(4, 3) = 24 ).

---

#### 5. **Binary Search Tree (BST)**

- **Definition**:
  - A Binary Search Tree is a binary tree where each node has at most two children.
  - For any node \( N ):
    - All elements in the left subtree of \( N ) are less than \( N ).
    - All elements in the right subtree of \( N ) are greater than \( N ).

![Binary Search Tree](../Archive/Attachment/Binary%20Search%20tree.png)
- **Key Operations**:
  - **Insertion**: Start from the root and insert the new node in the correct position such that the BST property is maintained.
  - **Search**: Start from the root and recursively move left or right, depending on whether the value is smaller or larger than the current node.
  - **Deletion**: There are three cases to handle:
    1. Node to be deleted is a leaf (has no children).
    2. Node to be deleted has one child.
    3. Node to be deleted has two children (replace with the in-order successor or predecessor).

- **Time Complexity**:
  - **Best Case** (balanced tree): \( O(\log n) )
  - **Worst Case** (unbalanced tree, e.g., a linked list): \( O(n) )

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
   - **Goal:** Determine if two functions \( f(x) ) and \( g(x) ) are identical.
   - **Approach:** 
     - **Randomized Testing:** 
       - Choose a random number \( r ) in the range \([0, 1000d]).
       - Evaluate the polynomial \( P(r) = f(r) - g(r) ).
       - If \( P(r) = 0 ), declare \( f(x) = g(x) ) (identical).
       - If \( P(r) \neq 0 ), declare \( f(x) \neq g(x) ) (not identical).
     - **Correctness:** The correctness of this randomized approach depends on how likely it is that the functions are identical or not. If the algorithm says "identical," it is checking whether \( r ) is a root of \( P(x) ). The probability of falsely declaring functions identical is minimized by testing multiple roots.

#### 2. **Polynomial Identity Testing (PIT):**
- This is the process where the algorithm checks if \( f(x) = g(x) ) by evaluating at random points. PIT is used in algorithms to probabilistically determine if two polynomials are identical without explicitly expanding them.

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
1. **Base Case**: The condition under which the recursion stops. It's crucial to prevent infinite recursion. For example, in calculating factorial, the base case is \( n = 0 ) or \( n = 1 ), where the function returns 1.

2. **Recursive Case**: The part of the function where it calls itself with a modified argument to reduce the problem size. 

#### Example: Factorial Function

function to calculate the factorial of a number \( n ):

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
  - **Probability of Correctness**: They have a certain probability of providing a correct answer. For instance, if the algorithm has a probability \( \alpha_1 ) of giving the correct answer and \( \alpha_2 ) of giving the wrong answer, \( \alpha_1 + \alpha_2 = 1 ).
  - **Optimization**: By optimizing the randomized algorithm (e.g., using more samples or better random choices), it is often possible to increase the probability of obtaining a correct result and potentially achieve better performance than some deterministic algorithms.
  - **Efficiency**: In practice, randomized algorithms can be more efficient than deterministic ones, especially for complex problems where deterministic approaches are too slow or infeasible.

#### 3. **Quantum Algorithms**
- **Definition**: Algorithms that leverage principles of quantum mechanics to solve problems, using quantum bits (qubits) and quantum operations.
- **Characteristics**:
  - **Quantum Superposition**: Quantum algorithms can process a superposition of states, allowing them to explore many possible solutions simultaneously.
  - **Interference**: Quantum algorithms use interference to amplify the probability of correct answers and cancel out incorrect ones.
  - **Probability of Correctness**: Quantum algorithms can be designed to reduce or eliminate the probability of incorrect answers by strategically using quantum operations. For example, quantum algorithms can cancel out the probability of incorrect answers (\( \alpha_2 )) and ensure that the output is correct with high probability (\( \alpha_1 )).
  - **Example**: Shor's algorithm for factoring large numbers and Grover's algorithm for searching an unsorted database are examples where quantum algorithms offer significant speedups over classical algorithms.
The discussion in your lecture about checking if a number is prime and the use of the square root of \( N ) in algorithms is quite relevant in number theory and algorithm design. Here's a structured note on this topic:

---

## Prime Number Testing and the Square Root Method

#### **1. Prime Number Testing**

**Definition**: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

**Goal**: To determine if a given number \( N ) is prime.

#### **2. Basic Approach**

- **Direct Method**: Check if \( N ) is divisible by any number from 2 up to \( N-1 ). If it is, \( N ) is not prime. This approach is inefficient for large numbers.

#### **3. Optimized Approach Using Square Root**

**Concept**: A more efficient method involves checking divisibility only up to \( $$\sqrt{N} $$). The rationale is based on the following observation:

- **Observation**: If \( N ) is divisible by some number \( d ), then \( N = d \times k ), where \( k ) is also a divisor. If both \( d ) and \( k ) were greater than \( \sqrt{N} ), their product \( d \times k ) would be greater than \( N ). Hence, at least one of these divisors must be less than or equal to \( \sqrt{N} ).

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
1. **Handle Small Cases**: Check if \( N ) is less than 2. If so, it is not prime.
2. **Check Divisibility**:
   - For each integer \( i ) from 2 up to \( \sqrt{N} ):
     - If \( N ) is divisible by \( i ), then \( N ) is not a prime.
   - If no divisors are found in this range, \( N ) is a prime.

# ADS - 09

### 1. **Correctness of Algorithms**
   - **Correctness** refers to proving that an algorithm produces the correct output for all valid inputs.
   - **Correctness proof** usually involves:
     1. **Partial correctness**: Showing that if an algorithm terminates, it produces the correct output.
     2. **Termination**: Showing that the algorithm always terminates.

### 2. **Loop Invariants**
   - A **loop invariant** is a property or condition that holds true before and after each iteration of a loop.
   - It is crucial in proving the correctness of algorithms, especially those involving loops.

### 3. **Algorithm for Finding Maximum Element in a List**
   - You described an algorithm to find the maximum element in a list `L` of positive distinct integers.
   - The algorithm iterates through the list, updating a variable `Max` whenever a larger element is found.

#### **Pseudocode:**
```python
Max = L[0]  # Initialize Max with the first element of L
for i in range(1, len(L)):
    if L[i] > Max:
        Max = L[i]
# At the end of this loop, Max contains the maximum element in L
```

### 4. **Correctness Proof Using Loop Invariants**
   - **Invariant:** A statement that is true before and after each iteration of the loop.
   - **INV:** This refers to the loop invariant in your context.

#### **Invariants for Maximum Algorithm:**
   - **Invariant 1:** For all `j` in `[0, i]`, `L[j] <= Max`.
     - **Meaning:** The variable `Max` always contains the largest value among the elements seen so far.
   - **Invariant 2:** `Max >= 0`.
     - **Meaning:** Since all elements are positive, `Max` is always non-negative.

#### **Proof:**
   - **Initialization:** Before the loop starts, `Max` is set to `L[0]`, so the invariant holds because the first element is the largest seen so far.
   - **Maintenance:** If the invariant is true before iteration `i`, then after the loop body executes, it remains true. If `L[i] > Max`, then `Max` is updated, so it still holds.
   - **Termination:** When the loop ends, the invariant and the loop condition imply that `Max` contains the largest element in the list.

#### **Correctness Statement:**
   - For all `i` in `[0, |L|-1]`, `L[i] <= Max`. This ensures that the algorithm correctly finds the maximum element. for any positive max no this statement will give correct answer
$$
\forall i \; \epsilon [0,\dots,|L|] 
$$$$ 
L[i]\leq max
$$
$$(\forall j \; \epsilon [0 \dots i+1] , L[j] \leq max) \cap (P)
% For All j from 0 to i+1 \left( i is index max will be equal to or greater than them )
$$$$
P = (\exists K|max=L[K])
$$
### 5. **Insertion Sort Algorithm**
   - **Insertion Sort** works by building a sorted section of the list one element at a time.

#### **Pseudocode:**
```python
for i in range(1, len(L)):
    key = L[i]
    j = i - 1
    while j >= 0 and L[j] > key:
        L[j + 1] = L[j]
        j -= 1
    L[j + 1] = key
```

#### **Correctness Proof Using Invariants:**
   - **Invariant:** After the `i`th iteration, the subarray `L[0:i+1]` is sorted.
   - **Proof:**
     - **Initialization:** Before the first iteration, the subarray `L[0:1]` is trivially sorted.
     - **Maintenance:** If the subarray `L[0:i]` is sorted before the iteration, the insertion of `L[i]` into the correct position within this subarray maintains the sorted order of `L[0:i+1]`.
     - **Termination:** When the loop terminates, the entire array `L[0:|L|-1]` is sorted.

### 6. **Correctness Statement for Insertion Sort:**
   - For all `i` in `[0, |L|-2]`, `L[i] <= L[i+1]`. This ensures that the list is sorted in non-decreasing order.

### **Summary:**
- **Loop Invariants** are key to proving the correctness of iterative algorithms.
- For the maximum element algorithm, the loop invariant ensures that `Max` is always the largest value encountered so far.
- For **Insertion Sort**, the invariant guarantees that the sorted portion of the list remains sorted after each insertion.

# ADS - 10

## Abstract Data Types (ADT) vs. Normal Data Types

### What is an Abstract Data Type (ADT)?
- **Abstract Data Types** are not concerned with how data is stored or implemented, but rather with what operations can be performed on data.
- Examples of Abstract Data types: Stack, Queue, List, etc.

### Normal Data Types
- **Normal Data Types** (like int, float, char) are concrete data types that are defined by a specific way of storing data in memory.

### Struct and Object (Not Abstract Data type)
- **Struct**: A way to group different types of data together in a single unit. Not considered an abstract data type.
- **Object**: A more complex data structure that includes both data and methods to operate on the data. Also, not considered an ADT.

## Arrays, Linked Lists, and Trees

### Arrays
- A **linear data structure** that stores elements in contiguous memory locations.
- Easy to access elements by index but resizing is difficult.

### Linked Lists
- A **linear data structure** where elements are stored in nodes, and each node points to the next one.
> We have to Linearly search & can't access specific index directly like array
- Easier to insert and delete elements compared to arrays, but access by index is slower.

### Trees
- A **non-linear data structure** with a root node and children nodes forming a hierarchy.
- Efficient for operations like search, insert, and delete.

## Binary Search Tree (BST)
- A tree where each node has at most two children, with the left child being less than the parent and the right child being greater.
- Allows efficient searching, but can become unbalanced.

## AVL Tree
- A self-balancing binary search tree where the height difference between the left and right subtree (balance factor) is at most 1.
- Ensures O(log N) time complexity for search, insert, and delete operations.

## Red-Black Tree
- A type of self-balancing binary search tree with an additional property of node colors (red or black).
- Ensures that the tree remains balanced and provides O(log N) time complexity for search, insert, and delete operations.
### Red-Black Tree Properties

1. Binary Search Tree Property
2. All nodes are either Red Black Colour
3. Root is always black
4. All leaves are black
5. Red nodes can have only black children & black nodes can have Black/Red children
6. For any node, the number of black nodes to any leaf is same

![Red Black Tree](../Archive/Attachment/Red-Black%20Tree.png)

## Height of Trees
- For a balanced binary search tree with **N nodes**, the desired height is **O(log₂N)**.
- Height plays a crucial role in determining the efficiency of operations in a tree.

### Example: Height Visualization

Consider a tree with **N = 8 nodes**.

The height **H** of a balanced binary tree with **N nodes** is:

$$[ H = \log_2 N = \log_2 8 = 3 ]$$

Here’s a simple diagram to visualize this:

```plaintext
       4
      / \
     2   6
    / \ / \
   1  3 5  7
            \
             8
```

- **Height = 3**, which matches the formula **O(log₂N)**.

# ADS - 11

## Overview

A Red-Black Tree (RB Tree) is a type of self-balancing binary search tree with specific properties that ensure the tree remains approximately balanced, making operations efficient.

## Key Properties

### Height of Red-Black Tree

- **Claim**: The height of an RB Tree is \( O(\log N) ), where \( N ) is the number of nodes. This property ensures that operations like insertion, deletion, and search are efficient due to the logarithmic height.

### Black Height

- **Definition**: The black height of a node \( x ) is the number of black nodes along the path from \( x ) to the leaves, excluding \( x ) itself. It helps in maintaining the balance of the tree.
  
- **Subclaim**: The number of nodes in the subtree of node \( x ) is at least \( 2^{\text{bh}(x)} - 1 ), where \( \text{bh}(x) ) is the black height of node \( x ). This provides a lower bound on the number of nodes based on the black height.

### Inductive Step

- **Height Calculation**:
  - For a Red-Black Tree of height \( h ), the number of nodes is \( \geq 2^{h/2} - 1 ). This is derived from the property that the minimum black height ensures a certain number of nodes.
  - Therefore, \( h = O(\log n) ), and \( \log n \geq 2^{h/2} - 1 ). This confirms that the tree height grows logarithmically with the number of nodes.

### Rotations

- **Left Rotation**:
  - **Purpose**: To maintain balance in the tree by rotating a node to the left.
  - **Procedure**: 
    1. Make the right child of the node the new root.
    2. The left child of the new root becomes the right child of the original root.
    3. Update the original root’s right child to be the new root.

- **Right Rotation**:
  - **Purpose**: To maintain balance by rotating a node to the right.
  - **Procedure**: 
    1. Make the left child of the node the new root.
    2. The right child of the new root becomes the left child of the original root.
    3. Update the original root’s left child to be the new root.

### Red-Black Tree Code Overview

- **Note**: The code provides hints on operations but is not complete. Important operations include rotations and color adjustments to maintain tree properties.

# ADS - 12

## Invariants

1. **Red Node Property**:
   - **Invariant**: If a node \( z ) is red, then its parent \( z.p ) must be black. This rule ensures that no two red nodes can be adjacent, which helps in maintaining tree balance.

2. **Violation Cases**:
   - **Root Red**: The root of the tree should not be red. If \( z ) is the root and is red, it violates the Red-Black Tree property.
   - **Consecutive Red Nodes**: Both \( z ) and its parent \( z.p ) should not be red. This prevents the violation of the Red-Black Tree rules that would otherwise lead to imbalanced trees.

## Proof of Correctness

- **Proof**: The correctness of the Red-Black Tree relies on these invariants. Ensuring that these properties hold true during operations like insertion and deletion guarantees that the tree remains balanced and maintains its logarithmic height.
# ADS - 13

## Transplant Operation

- **Definition**: The `Transplant` operation replaces a node \( z ) with its child \( y ). This is crucial for maintaining the binary search tree property after a node is deleted.
  
- **Code Example**:
  ```python
  def transplant(root, u, v):
      if u.p is None:
          root = v
      elif u == u.p.left:
          u.p.left = v
      else:
          u.p.right = v
      v.p = u.p
      return root
```

## Deletion Cases

1. **Node with Two Children**:
    - **Procedure**: Replace node z with its right-side lowest value child.
    - **Reason**: The right-side lowest value child ensures the binary search tree property is preserved.
2. **No Left Child**:
    - **Procedure**: Replace z with its right child.
3. **No Right Child**:
    - **Procedure**: Replace z with its left child.

## Fixing Up
- **Red Leaf Deletion**: 
	- (y(child)=Red)
    - Directly maintains the Red-Black Tree properties without additional fixes.
- **Black Node Deletion**: 
	- (y(child)=Black & z(deletion node)=Black/Red)
    - If z is black and y is black, or z is red and y is black, adjustments are needed to maintain tree properties. This involves fixing the double black issues by rebalancing and recoloring nodes.


# ADS - 14

## Red-Black Tree Deletion Overview

Deletion in a Red-Black Tree involves removing a node while maintaining the Red-Black Tree properties. This includes handling various cases and adjusting the tree to ensure it remains balanced.

## Deletion Cases and Fixes

### Deleting a Node with Two Children

1. **Find the Successor**:
   - If the node \( z ) to be deleted has two children, find the node \( y ) with the smallest value in the right subtree of \( z ). This node \( y ) will replace \( z ).
   
2. **Transplant Operation**:
   - Replace \( z ) with \( y ). Since \( y ) will have at most one child (it may have a right child but not a left child), you need to adjust the tree by transplanting \( y ) into \( z )'s position.

3. **Fix-Up**:
   - After transplantation, if \( y ) was black, adjustments are needed to fix the Red-Black Tree properties. This may involve recoloring and performing rotations.

### Deleting a Node with Only One Child

1. **Simple Replacement**:
   - If the node \( z ) to be deleted has only one child, replace \( z ) with its child. Adjust the parent of \( z ) to point to \( z )'s child.

### Deleting a Node with No Children (Leaf Node)

1. **Direct Removal**:
   - Simply remove the leaf node. If the node is red, there are no additional adjustments required. If it is black, it may cause a double-black issue, which requires further adjustment.

## Fixing Double-Black Issues

### Double-Black Fix Algorithm

1. **Case 1: Sibling is Red**:
   - If the sibling \( s ) of the node being fixed is red, perform a rotation and recoloring to move the red sibling up and fix the double-black issue.

2. **Case 2: Sibling is Black and Sibling’s Children are Black**:
   - If the sibling \( s ) is black and both of \( s )'s children are black, move the double-black property up the tree and recolor \( s ) to red.

3. **Case 3: Sibling is Black and Sibling’s Children are Not Both Black**:
   - Adjust the tree based on the color of \( s )’s children. This often involves rotations and recoloring.

### Code for Deletion and Fix-Up

Here is a simplified Python code example for Red-Black Tree deletion and fixing up double-black issues:

```python
class Node:
    def __init__(self, data, color='red', left=None, right=None, parent=None):
        self.data = data
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

def delete_node(root, z):
    if z.left and z.right:
        y = minimum(z.right)
        z.data = y.data
        z = y

    x = z.left if z.left else z.right

    if x:
        x.parent = z.parent

    if not z.parent:
        root = x
    elif z == z.parent.left:
        z.parent.left = x
    else:
        z.parent.right = x

    if z.color == 'black':
        fix_double_black(root, x)

    return root

def fix_double_black(root, x):
    # Implement the fix-up logic for double-black nodes here
    pass

def minimum(node):
    while node.left:
        node = node.left
    return node

def rb_insert_fixup(root, z):
    while z != root and z.parent.color == 'red':
        if z.parent == z.parent.parent.left:
            y = z.parent.parent.right
            if y and y.color == 'red':
                z.parent.color = 'black'
                y.color = 'black'
                z.parent.parent.color = 'red'
                z = z.parent.parent
            else:
                if z == z.parent.right:
                    z = z.parent
                    left_rotate(root, z)
                z.parent.color = 'black'
                z.parent.parent.color = 'red'
                right_rotate(root, z.parent.parent)
        else:
            # Symmetric case
            pass

    root.color = 'black'

def left_rotate(root, x):
    # Implement left rotation logic here
    pass

def right_rotate(root, y):
    # Implement right rotation logic here
    pass
```

# ADS - 15

## Merge Sort

### Overview
Merge Sort is a stable, comparison-based sorting algorithm that follows the divide-and-conquer strategy. It divides the array into smaller subarrays, sorts each subarray, and then merges them to produce a sorted array.

### Steps

1. **Divide**: 
   - Split the array into two halves until each subarray contains a single element.

2. **Conquer**: 
   - Recursively sort each half.

3. **Combine**: 
   - Merge the sorted halves to create a single sorted array.

### Time Complexity
- **Best, Average, and Worst Case**: \( O(n \log n) )
- **Space Complexity**: \( O(n) ) due to the auxiliary space used for merging.

## Hashing

### Overview
Hashing is used to map data to a fixed-size value (hash value) to enable efficient data retrieval. It is commonly implemented using hash tables.

### Key Concepts

1. **Hash Function**:
   - A function that converts input data into a hash value. It should ensure a uniform distribution of data.

2. **Collision Handling**:
   - **Chaining**: Use linked lists to handle collisions by storing multiple items in the same hash slot.
   - **Open Addressing**: Find another open slot in the hash table (e.g., linear probing, quadratic probing).

3. **Load Factor**:
   - Ratio of the number of elements to the number of slots in the hash table. Higher load factors increase the chance of collisions.

### Time Complexity
- **Average Case**: \( O(1) ) for search, insert, and delete operations.
- **Worst Case**: \( O(n) ) in case of many collisions.

## Divide and Conquer

### Overview
Divide and Conquer is an algorithmic strategy that involves breaking a problem into smaller subproblems, solving each subproblem independently, and combining the solutions to solve the original problem.

### Steps

1. **Divide**:
   - Break the problem into smaller subproblems.

2. **Conquer**:
   - Recursively solve each subproblem. If the subproblems are small enough, solve them directly.

3. **Combine**:
   - Merge the solutions of the subproblems to get the solution to the original problem.

### Examples

- **Merge Sort**: Divides the array, sorts each half, and merges the sorted halves.
- **Quick Sort**: Chooses a pivot, partitions the array into elements less than and greater than the pivot, and recursively sorts the partitions.

### Time Complexity
- **Merge Sort**: \( O(n \log n) )
- **Quick Sort**: \( O(n \log n) ) on average, \( O(n^2) ) in the worst case.

# ADS - 16

## Finding Closest Pair of Points

### Overview
In a 2D plane with multiple points, the goal is to find the pair of points with the closest distance between them using the Divide and Conquer approach.

### Steps

1. **Divide**:
   - Divide the set of points into two halves: left and right.

2. **Conquer**:
   - Find the closest pair of points in the left half, \( d1 ).
   - Find the closest pair of points in the right half, \( d2 ).

3. **Combine**:
   - Determine the minimum distance between \( d1 ) and \( d2 ). Let’s denote this minimum distance as \( d ), where \( d = \min(d1, d2) ).

4. **Check Points Near the Center**:
   - Consider points within a vertical strip centered around the dividing line with width \( 2d ). Check if there are any pairs of points within this strip that are closer than \( d ).

5. **Pythagorean Check**:
   - The Pythagorean theorem helps in understanding the geometric constraints, **Distance Bound**: If you have a point p in the vertical strip and want to check distances to points in the same strip, the maximum distance to check (considering both x and y distances) would be bounded by $$\sqrt{ 2d^2 }$$​. This is why the theorem is referenced: it gives insight into how far apart points can be in a certain geometric configuration.

### Time Complexity
- The Divide and Conquer approach typically has a time complexity of \( O(n \log n) ) due to the efficient merging of results from divided sections.

## Matrix Multiplication

### Standard Matrix Multiplication

- For matrices \( A ) and \( B ) of size \( n \times n ):$$ [
  C = A \times B
  ]$$
  The time complexity of the standard matrix multiplication is $$( \Theta(n^3) )$$

## Matrix Multiplication for \(2 \times 2) Matrices

Given two \(2 \times 2) matrices \(A) and \(B):

$$[
A = \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}
\quad \text{and} \quad
B = \begin{pmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22}
\end{pmatrix}
]$$

The product \(C = A \times B) is calculated as follows:

$$[
C = \begin{pmatrix}
c_{11} & c_{12} \\
c_{21} & c_{22}
\end{pmatrix}
]$$

where each element of \(C) is computed by:

$$
\begin{aligned}
c_{11} &= a_{11}b_{11} + a_{12}b_{21} \\
c_{12} &= a_{11}b_{12} + a_{12}b_{22} \\
c_{21} &= a_{21}b_{11} + a_{22}b_{21} \\
c_{22} &= a_{21}b_{12} + a_{22}b_{22}
\end{aligned}
$$

### Recurrence Relation

- When dividing the matrices into smaller submatrices, the recurrence relation for matrix multiplication is:
$$  [
  T(n) = 8T(n/2) + C
  ]$$
  - **8T(n/2)**: This term accounts for the 8 recursive multiplications of n/2×n/2​ matrices.
  - where ( C ) is the cost of combining the results.

### Strassen's Algorithm

- Strassen's Algorithm improves the time complexity of matrix multiplication to $$( \Theta(n^{2.81}) )$$

#### Key Points

1. **Algorithm Overview**:
   - Strassen’s algorithm reduces the number of multiplications required by decomposing the matrix multiplication problem into smaller subproblems. (S1...S10)

2. **Formulas**:
   - Strassen’s algorithm uses 7 multiplications and 10 addition/subtraction operations to compute the product of two matrices. (P1...P7)

3. **Recurrence Relation**:
   - The recurrence relation for Strassen’s Algorithm is:    $$ [
     T(n) = 7T(n/2) + \Theta(n^2)
     ]$$
   - Here, ( \Theta(n^2) ) accounts for the addition and subtraction operations required.

### Computational Complexity

- **Matrix Multiplication**: The standard approach has ( \Theta(n^3) ) time complexity.
- **Strassen's Algorithm**: Reduces the time complexity to ( \Theta(n^{2.8}) ), making it more efficient for large matrices.

### Operations

- **Multiplication**: More computationally intensive and time-consuming. $$\theta (n)^3$$
- **Addition/Subtraction**: Less time-heavy compared to multiplication and requires simpler operations.$$\theta (n)^2$$
# ADS - 17

## Matrix Chain Multiplication (Dynamic Programming)

### Introduction
- **Matrix multiplication** involves multiplying two matrices to get a product matrix.
- Direct matrix multiplication has a cost measured by **scalar multiplications**.
- The goal is not to compute the matrix product but to find the optimal way to group matrices for multiplication to **minimize scalar multiplications**.

### Why Scalar Multiplication is Important?
- **Scalar multiplication** refers to multiplying individual elements during matrix multiplication.
- The number of scalar multiplications needed depends on how the matrices are grouped.
- By reordering or regrouping, we can reduce the number of scalar multiplications significantly.

### Problem: Matrix Chain Order
- Given a sequence of matrices A1, A2, ..., An with dimensions p0 x p1, p1 x p2, ..., pn-1 x pn.
- We need to find an order of matrix multiplication that minimizes the total scalar multiplication.

#### Key Formula for Scalar Multiplication Cost:
For matrices Ai (of dimension pi x pi+1) and Ak (of dimension pk x pk+1), the scalar multiplication cost for multiplying two matrices is:
$$Cost = m_ik + m_kj + pi * pk * pj$$

Where:
- m_ik = Minimum number of scalar multiplications for matrices from i to k.
- m_kj = Minimum number of scalar multiplications for matrices from k+1 to j.
- pi * pk * pj is the cost of multiplying the two matrices.

## Dynamic Programming Approach
1. **Subproblem**: Define m[i, j] as the minimum number of scalar multiplications required to multiply matrices Ai to Aj.
2. **Recurrence relation**:

$$m[i, j] = min(i <= k < j) { m[i, k] + m[k+1, j] + p(i-1) * pk * pj }$$

3. **Base Case**: When multiplying a single matrix, i.e., m[i, i] = 0, since no multiplication is needed.
4. **Optimal Parenthesization**: Store the optimal index k to split the matrix multiplication for the minimal cost.
5. **Final Solution**: m[1, n] gives the minimum number of scalar multiplications for multiplying matrices from A1 to An.

## Time Complexity
- **Time Complexity**: Theta(n^3)
- We calculate the minimum cost for multiplying matrices in all possible ways using dynamic programming tables.
- For n matrices, there are n^2 subproblems, each requiring O(n) work, resulting in O(n^3) overall complexity.

## Example
For matrices with dimensions:
- A1 = 10 x 30, 
- A2 = 30 x 5,
- A3 = 5 x 60.

Find the optimal parenthesization order that minimizes scalar multiplications.

### Summary:
1. Identify the matrix dimensions.
2. Set up the cost matrix and find the minimal scalar multiplications using dynamic programming.
3. Solve the subproblems using the recurrence relation.
4. Use dynamic programming tables to store intermediate results for efficiency.

