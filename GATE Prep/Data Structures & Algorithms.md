## Sorting Algorithms

Sorting algorithms are used to arrange the elements of a list or array in a certain order, typically in ascending or descending order. Here are some common sorting algorithms along with a brief explanation of each:

### 1. Bubble Sort
Bubble Sort is a simple comparison-based algorithm where each pair of adjacent elements is compared and swapped if they are in the wrong order. This process is repeated until the list is sorted.

**Steps:**
- Compare the first two elements, swap if necessary.
- Move to the next pair, and repeat until the end of the list.
- Repeat the entire process for the next elements until no more swaps are needed.

**Time Complexity:** O(n²)

### 2. Selection Sort
Selection Sort works by repeatedly finding the minimum element from the unsorted part of the list and placing it at the beginning.

**Steps:**
- Find the minimum element in the list.
- Swap it with the first element.
- Repeat the process for the remaining elements, starting from the second position.

**Time Complexity:** O(n²)

### 3. Insertion Sort
Insertion Sort builds the final sorted array one item at a time. It picks the next element and inserts it into the correct position in the already sorted part of the list.

**Steps:**
- Start with the second element, compare it with the first, and insert it in the correct position.
- Move to the third element and repeat the process for all elements.

**Time Complexity:** O(n²)

### 4. Merge Sort
Merge Sort is a divide-and-conquer algorithm that splits the list into halves, recursively sorts each half, and then merges the sorted halves back together.

**Steps:**
- Divide the list into two halves.
- Recursively sort each half.
- Merge the sorted halves to produce the sorted list.

**Time Complexity:** O(n log n)

### 5. Quick Sort
Quick Sort is another divide-and-conquer algorithm that selects a 'pivot' element and partitions the list into two sub-arrays, according to whether the elements are less than or greater than the pivot. The sub-arrays are then sorted recursively.

**Steps:**
- Choose a pivot element.
- Partition the array so that elements less than the pivot are on the left, and elements greater than the pivot are on the right.
- Recursively apply the same logic to the sub-arrays.

**Time Complexity:** O(n log n) on average, O(n²) in the worst case.

### 6. Heap Sort
Heap Sort involves building a binary heap from the list and then repeatedly extracting the maximum element from the heap and rebuilding the heap until all elements are sorted.

**Steps:**
- Build a max-heap from the list.
- Swap the root (maximum value) with the last element of the heap.
- Reduce the heap size and heapify the root element.
- Repeat until the heap is empty.

**Time Complexity:** O(n log n)