# **Basic Info**

- Most loved Programming language in Systems.
- Generally used High level Simplicity & Low level performance
- Exceptionally good when performance is critical and focus.
- No Garbage collector but do those task using Ownership and Garbage Collector .
- Immutable variable by default, Automatically drop value when no owner.
- Rust work using Cargo.

## **Overview**

Rust is a modern, high-performance, statically typed programming language designed for:
- **Memory Safety**: No null or dangling pointers.
- **Concurrency**: Built-in tools for safe concurrent programming.
- **Speed**: Comparable to C and C++ with zero-cost abstractions.

### **Key Features**
1. **Ownership and Borrowing**: Efficient memory management without garbage collection.
2. **Concurrency**: Prevents data races at compile time.
3. **Performance**: Ideal for system-level programming and performance-critical applications.
4. **Tooling**: Comes with `Cargo`, a package manager and build tool.
5. **Ecosystem**: Rich libraries available via `crates.io`.

### **When to Use Rust**
- System programming.
- Performance-critical applications.
- Safe replacements for C/C++ in legacy systems.
- Concurrent and parallel applications.

---

## **Code Examples**

### **1. Hello, World!**

```rust
fn main() {
    println!("Hello, world!");
}```

```rust
fn main() {
    let s = String::from("hello"); // Ownership of the string
    takes_ownership(s);           // Ownership is moved
    // println!("{}", s);         // Error: `s` is no longer valid

    let x = 5;
    makes_copy(x);                // x is copied, so it's still valid
    println!("{}", x);
}

fn takes_ownership(some_string: String) { //Taking string as its pass to fucntion ownership transfer
    println!("{}", some_string);
}

fn makes_copy(some_integer: i32) { //Making copy as it's using i32
    println!("{}", some_integer);
}

```