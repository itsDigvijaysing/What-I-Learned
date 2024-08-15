\Links: [IIT Hyderabad](IIT%20Hyderabad.md), [Data Structures & Algorithms](../GATE%20Prep/Data%20Structures%20&%20Algorithms.md)

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
- Algo A: 