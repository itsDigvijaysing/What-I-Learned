### Tags: #Cpp 
## Basics

C++ inherits most of its code style from C language, but both are very different from each other. 
Let's consider an example:

```cpp
// This is a comment

/*
  This is a block code commented.
*/

#include <cstdio> //<- Libraries
#include <iostream>

using namespace std; //<- scope to identifiers

int main(int argc, char ** argv) {
  puts("hello"); // this statement outputs hello with a new line
  printf("hello\n"); // this is similar to puts but doesn't end with new line
  cout << "hello\n"; // more complex way to output without new line
  return 0; // 0 means success
}
```

A C++ program can also be written like this (though I wouldn't recommend it):

```cpp
#include <cstdio>
using namespace std;

int
main
(
int
argc,
char
**
argv) {
puts("
hello")
  ;
  return 0;
}
```


**Things to remember**

1. A statement should always end with `;`.
2. `#Include` should always be in single line without any space followed by `<>` or `""`.
3. Libraries, We generally use `<iostream>` , It's good library but it doesn't include other extra functionality of c++, so in competitive environment people generally use `<bits/stdc++.h>` Library it include almost all c++ libraries functionality, It's process heavy but helpful as we don't need to include libraries everytime. (It's supported for `gcc` & `clang with libstdc++` integrated.)

### Identifiers

* These identifiers cannot conflict with C++ 86 keywords (which includes 11 tokens)
  
| signed                      | alignas (since C++11)   | explicit               |
| --------------------------- | ----------------------- | ---------------------- |
| sizeof                      | alignof (since C++11)   | export(1)              |
| static                      | and                     | extern                 |
| static_assert (since C++11) | and_eq                  | FALSE                  |
| static_cast                 | asm                     | float                  |
| struct                      | auto(1)                 | for                    |
| switch                      | bitand                  | friend                 |
| template                    | bitor                   | goto                   |
| this                        | bool                    | if                     |
| thread_local (since C++11)  | break                   | inline                 |
| throw                       | case                    | int                    |
| TRUE                        | catch                   | long                   |
| try                         | char                    | mutable                |
| typedef                     | char16_t (since C++11)  | namespace              |
| typeid                      | char32_t (since C++11)  | new                    |
| typename                    | class                   | noexcept (since C++11) |
| union                       | compl                   | not                    |
| unsigned                    | concept (concepts TS)   | not_eq                 |
| using(1)                    | const                   | nullptr (since C++11)  |
| virtual                     | constexpr (since C++11) | operator               |
| void                        | const_cast              | or                     |
| volatile                    | continue                | or_eq                  |
| wchar_t                     | decltype (since C++11)  | private                |
| while                       | default(1)              | protected              |
| xor                         | delete(1)               | public                 |
| xor_eq                      | do                      | register               |
|                             | double                  | reinterpret_cast       |
|                             | dynamic_cast            | requires (concepts TS) |
|                             | else                    | return                 |
|                             | enum                    | short                  |

* Identifiers are case sensitive.

### Pair

-  In CPP we can store key value pair using pair<int, int> it will create key value pair.
```cpp
pair<int, pair<int,int>> p = {1,{2,3}}
p.first -> 1
p.second.second -> 3
p.second.first -> 2
```
### Defining Variables
Identifiers (or variables) can be initialized by using the following syntax:

```
DataType VariableName = "String" or number;
```

You can also define a read-only variable or a constant in C++ by using the keyword `const`.
String are nothing but array of char, so if we want to change something just write `s[index]=change` & String will be changed in that way.

#### Primitive Data Types
| Type      | Description                       | Example         |
|-----------|-----------------------------------|-----------------|
| `int`     | Integer                           | `42`, `-1`      |
| `float`   | Single-precision float            | `3.14f`         |
| `double`  | Double-precision float            | `3.14159`       |
| `char`    | Single character                  | `'a'`, `'Z'`    |
| `bool`    | Boolean (`true`/`false`)          | `true`, `false` |
| `void`    | No data (used in functions)       | N/A             |
| `wchar_t` | Wide character                    | L`'A'`, L`'Ω'`  |

#### Modifiers

| Modifier   | Description             |
| ---------- | ----------------------- |
| `signed`   | Positive & negative     |
| `unsigned` | Non-negative only       |
| `short`    | Smaller range (2 bytes) |
| `long`     | Larger range            |

#### Derived Data Types
- **Arrays**: `int arr[10];`
- **Pointers**: `int *ptr;`
- **References**: `int &ref = var;`
- **Functions**: `int sum(int a, int b);`

#### User-Defined Data Types
| Type      | Description                                |
|-----------|--------------------------------------------|
| `struct`  | Group different types                      |
| `class`   | Object-oriented programming                |
| `enum`    | Named integer constants                    |
| `union`   | Multiple types sharing same memory         |
| `typedef` | Alias for a data type                      |

### Pointer and Reference

A variable can be called in two ways; `call by value` and `call by reference`.

A pointer and reference are a type of data type, which is commonly used in C/C++ programming. It is a very powerful and confusing concept. It will take some time to understand.

* A pointer can take reference of another variable or a real value

Lets understand how identifiers work. When we say `int a = 10;`, an integer variable `a` has a value of `10` in the memory. When we say `int b = a;`, an integer variable `b` has a copy of variable `a`

```cpp
int a = 10;
int b = a; // b = 10
```

**Pointers**

So, When I say `int *c = &a`, it means that pointer `c` points to the reference of `a`.

```cpp
int a = 10;
int *b = &a;
printf("%d\n", *b); // this will print the reference value of a, which is 10
```

**Reference**

References are the address of value in the memory. The pointer points to this address while calling.

* A reference can only call a variable which is already initialized.

So, when I say `int &d = b` the address if `b` is stored in `d`.

```cpp
int b = 20;
int &d = b;
printf("%d\n", d);
```

### Arrays and Strings

There are two types of Arrays and String in C++, one using C style Arrays & String and the second one using Standard Library (STL), which will be covered later.

**Arrays**

The syntax of a C-based array is

```cpp
int a[5] = {1,2,3,4,5}; // array[SizeOfArray] = {'contents'};
printf("%d\n", a[0]); // 1
```
**Strings**

A string is an array of characters terminated string or also called as null terminated string. A string is always terminated with 0.

```cpp
char a[6] = {'a', 'b', 'c', 'd', 'e', 0};
printf("%s\n", a); // abcde
```

### Vector

- Container Dynamic in Nature (When no size is known)
- we can push_back in vector.
```cpp
vector<int> v;
v.push_back(2);
v.emplace_back(3); //Same work as push_back but little bit faster

vector<int> v(5,100); //Container of Size 5 with every value 100 [100,100,100,100,100]
vector<int> new_v(v); //Full copy of v container
```

#### Vector Iterator

- `v.begin()` returns an iterator pointing to the first element (i.e., the address of the first element).
- `v.rbegin()` reverse to v.begin(), it give last element address.
- `v.end()` points past the last element (not the last element itself)(Not at last, its next to the last element address).
- `v.rend()` reverse to v.end(), it give address of the element before first element.
- `*it` dereferences the iterator, accessing the actual data at that memory location.
- `it++` moves to the next element, `it--` moves to the previous.
- `vec.insert(it, value)` inserts before `it`, `vec.erase(it)` removes at `it`.

#### Vector loop

- auto = automatically assign the datatype.

```cpp
for(auto it=v.begin();it!=v.end();it++){
	cout<<*(it)<<endl;
}

for(auto it:v){
	cout<<it<<endl;
}
```

#### Vector Insert & Delete

- **`v.insert(it, value)`** → Inserts `value` at `it`, shifts elements right.
- **`v.insert(it, n, value)`** → Inserts `n` copies of `value` at `it`.
- **`v.insert(it, range_start, range_end)`** → Inserts elements from another range.
- **`v1.swap(v2)`** → Swaps contents of `v1` and `v2` in `O(1)`.

- **`v.erase(it)`** → Removes element at `it`, shifts remaining elements left.
- **`v.erase(it1, it2)`** → Removes elements in the range `[it1, it2)`.
- **`v.clear()`** → Removes all elements, size becomes `0`, but capacity remains.
- **`v.pop_back()`** → Removes the last element, reducing size by `1`.
### Conditions

There are two ways to use conditional operators.

1. Traditional conditional operator.
2. Ternary conditional operator.

**Traditional conditional operator**

`if..else..` are the common type of conditional statements.

```cpp
int a = 10;
int b = 20;

if (a > b) {
  puts("a>b");
} else {
  puts("b>a");
}
```

**Ternary conditional operator**

Its a one liner conditional operator

```cpp
int a = 10;
int b = 20;

printf("%d\n", a > b ? a : b); // if a is greater than b, print a else print b
```

### Switch Case

It is a conditional statement, which requires an expression which should satisfy a condition. If a condition is not satisfied, then it jumps to `default`. An expression should always be a constant of integer or a character. Syntax looks something like this

```cpp
switch (/* expression */) {
  case /* value */:
    /* statement */
}
```

### While Loop

There are two types of `While` loop in C++

1. `While` loop
   Only execute if Condition is satisfied

  ```cpp
  while (/* condition */) {
    /* code */
  }
  ```

2. `do.. While..` loop
   It get executed 1st time irrespective of the condition.

  ```cpp
  do {
    /* code */
  } while(/* condition */);
  ```

### For Loop

For loop is like `while` loop but with some extra features

```cpp
for (size_t i = 0; i < count; i++) {
  /* code */
}
```
### Range base `For` loop

Starting from C++11, we can use range based `For` loop

```cpp
for (type var1 : var2) {
  /* code */
}

for (int num : numbers) {
	cout << num << " "; // Output each number
}
```

### Using stdout

C++ also has a way to used object oriented way of printing out contents to the terminal/command prompt. So far we have used `printf` and `puts`.

```cpp
std::cout << "Hello World!" << std::endl;
```

The above code shows a bitwise stream of string to `cout`. The `<<` shows left shift of the content.

Creating a compiled version of `cout` uses a lot of resources when compared to `puts` or `printf`, this is because to compile `cout` the whole standard library - `STL` - is copied.

## Functions

A function can be defined as a block of code that is separate from the existing code; that is all the variables used in a function would only belong to that particular function. For example (pseudo code):

```cpp
int sum (int a, int b){
  return a + b;
}

int main(){
	int a = 10;
	int b = 20;
	c = sum(a, b);
	printf("%d\n", c);
return 0;
}
```

From the above the variables `a` and `b` in function `sum()` are different from the initialized variable `a` and `b`.

This particular type of function is call `call by value` (Send Copy) function. Another type of function is called as the `call by reference` or sometimes called as the `call by address`. For example (pseudo code):

```cpp
int sum (int *a, int *b){
  return *a + *b;
}

int main(){
	int a = 10;
	int b = 20;
	
	c = sum(&a, &b);
	printf("%d\n", c);
	return 0;
}
```

#### 1. Void Function 
- **Definition**: A function that does not return any value. It performs an operation but does not produce an output that can be used by the caller. 
#### 2. Return Function 
- **Definition**: A function that returns a value of a specified type. The return value can be used by the calling code for further processing or calculations. 
#### 3. Parameterized Function 
- **Definition**: A function that takes parameters (arguments) as input. These parameters allow the function to operate on different data each time it is called, making it more flexible. 
#### 4. Non-Parameterized Function 
- **Definition**: A function that does not take any parameters. It operates independently of external input and typically performs a fixed operation.

### Defining a function

In C++, a function should be declared first before calling it. That is:

```cpp
void name(/* arguments */) {
  /* code */
}

int main(int argc, char const *argv[]) {
  name()
  return 0;
}
```

C++ will not compile if the function being called is written after the main function.

To overcome this problem, we have something called `Forward Declaration`. For example:

```cpp
void name(/* arguments */);

int main(int argc, char const *argv[]) {
  name()
  return 0;
}

void name(/* arguments */) {
  /* code */
}
```

`void name(/* arguments */);` is know as `Forward Declaration` or prototype of `name()`

The common way to do `Forward Declaration` is to put the prototype in a header file. For example:

`3_1_3_Function_Header.cpp`

```cpp
#include "3_1_3_Function_Header.h"

int main(int argc, char const *argv[]) {
  name()
  return 0;
}

void name(/* arguments */) {
  /* code */
}
```

`3_1_3_Function_Header.h`

```cpp
#ifndef 3_1_3_Function_Header_h
#define 3_1_3_Function_Header_h

void name(/* arguments */);

#endif
```

### Passing values to a function

There are two two ways to pass values to a function

1. Pass by Value
2. Pass by Reference

**Pass by value:**

When you pass a value to a function, a copy of that value is stored in the argument.

```cpp
void sum(int c, int d) {
    printf("%d\n", c + d);
}

int main(int argc, char const *argv[]) {
    int a = 10;
    int b = 20;

    sum(a,b);
    return 0;
}
```

**Pass by Reference:**

We will talk more about pointers in the coming chapters. In C/C++ (but not limited to theses languages), when you create a variable a memory is allocated to that variable, this memory space has an address (location of it), so the reference here means we are sending the address of the variable rather than the variable itself.

For example, let us consider `int a = 10;`, which means an integer variable `a` has a value of `10` if you convert this in a diagrammatically you will get the following:

```
int a = 10;
----------
| a | 10 |  --> 123456
----------
```

The number `123456` is the address/location of integer `a` in the memory. When passing the value by reference you send this address, that means you do not create extra space for data; you just use what you have.

```cpp
void sum(int *a, int *b){
    printf("%d\n", *a+*b); // *a and *b pointing to the address given to them.
}

int main(int argc, char ** argv) {
    int a = 10;
    int b = 20;
    sum(&a,&b); // address of a and b
    return 0;
}
```
or
```cpp
void sum(int &a){ //Will take address of variable 'a' (argument)
	a=a*2; //This will modify the original variable 'a' (reference)
	cout<<a<<endl;
}

int main(){
	int a;
	cin>>a;
	sum(a);
	cout<<a<<endl;
	return 0;
}
```

> Array always go with reference by default.

There is one problem with pointers in C/C++, that is if you change the contents of the address in `sum()` function you will change the value of the variable. For example If we add a new integer `a=30` or `*a=30` variable to `sum()`

```cpp
void sum(int &a, int &b){
    a = 30;
    printf("%d\n", a+b);
}

// or

void sum(int *a, int *b){
    *a = 30;
    printf("%d\n", *a+*b);
}

```

The value of `a` is completely changed, for this not to happen we will have to use a keyword called `const`.

```cpp
void sum(const int &a, const int &b){
    a = 30;
    printf("%d\n", a+b);
}

// or

void sum(const int *a, const int *b){
    *a = 30;
    printf("%d\n", *a+*b);
}

```

### Automatic variables vs. Static variables

**Automatic variable**

By default, C++ uses automatic variables in every function. Whenever the function is called the variable local to it is initialized on a stack. For example

```cpp
void name() {
    int a = 10;
    printf("%d\n", a);
    a = 30;
}

int main(int argc, char const *argv[]) {
    name();
    name();// this will always print the same thing
    return 0;
}
```

**Static variable**

Unlike automatic variables Static variables do not get created on every function call, they just use whatever was previously defined. Don't forget to use `const` if you don't want to change the value.

### Return a function call

To return a function, we would have to type in the return type and use the keyword `return` at the end of the function. For example:

```cpp
int number(){
    return 10;
}

int main(int argc, char const *argv[]) {
    printf("%d\n", number());
    return 0;
}
```

### Function pointer

You can call a function by pointing to it, the same way you point to a variable. The only difference is that the data type of the function should match with the data type of the function pointer. For example

```cpp
	void name( {
		puts("hello");
	}

int main(int argc, char const *argv[]) {
	void (*function_pointer)() = name;
	function_pointer();
	return 0;
}
```

### Overloading function names

In C++ you can have multiple functions with the same name, but the signature (data type) should be same all over the function.

### Overloading operators with function

In C++ you can change the definition of the following 38 operators:

<table>
  <tr>
    <td>+</td>
    <td>-</td>
    <td>*</td>
    <td>/</td>
    <td>%</td>
    <td>^</td>
    <td>&amp;</td>
    <td>|</td>
    <td>~</td>
    <td>!</td>
    <td>=</td>
    <td>&lt;</td>
    <td>&gt;</td>
    <td>+=</td>
    <td>-=</td>
  </tr>
  <tr>
    <td>*=</td>
    <td>/=</td>
    <td>%=</td>
    <td>^=</td>
    <td>&amp;=</td>
    <td>|=</td>
    <td>&lt;&lt;</td>
    <td>&gt;&gt;</td>
    <td>&gt;&gt;=</td>
    <td>&lt;&lt;=</td>
    <td>==</td>
    <td>!=</td>
    <td>&lt;=</td>
    <td>&gt;=</td>
    <td>&amp;&amp;</td>
  </tr>
  <tr>
    <td>||</td>
    <td>++</td>
    <td>--</td>
    <td>,</td>
    <td>-&gt;*</td>
    <td>-&gt;</td>
    <td>( )</td>
    <td>[ ]</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

That means an addition operator can be turned into multiplication operator.

### Variable number of arguments

In C++ you can have multiple arguments given to a function, this can be achieved by adding `...` in the function arguments space.

There are four macros that needs to be called when using a variable arguments:

* *va_list*: `va_list fa` is used as a parameter.
* *va_start*: `va_start(ap, parameter)` initialize a variable argument list.
* *va_arg*: `va_arg(ap, type)` gets the next available argument of a data type.
* *va_end*: `va_end(ap)` Ends using variable argument list

### Recursive function

In C++ you can call a function itself. For example:

```cpp
int main(int argc, char const *argv[]) {
	main();
	return 0;
}
```

These types of functions are called recursive functions. These functions as an alternate to For loops.

## Preprocessors

The preprocessors are used to process the code before sending it to the compiler. The most common way is the file inclusion using `#include <>`. You can also use macro preprocessors by using `#define NUMBER 1`, these acts like a string substitution.

When you open a `.h` the contents of the file you often see looks something like this:

```cpp
#ifndef main_h
#define main_h

void function_name();

#endif /* main_h */
```

They are called as an "include guard" which checks for inclusion.

Another type of preprocessor is used by using `#pragma` that are used (or targeted) for specific compilers and architectures.

### Macro constants

You can define a macro constant by using `#define macro`. For example:

```cpp
#define NUMBER 1

int main(int argc, char const *argv[]) {
	printf("%d\n", NUMBER);
	return 0;
}
```

When the above code is compiled the NUMBER is replaced by a literal value before the code reaches to the compiler. At this point you cannot get its address or use pointers.

### Including a file

To include a file in a C++ file you would have to use `#include "file_name.h"`. This will place all the contents in the `cpp` before the code is sent to the compiler.

### Conditions in preprocessor

Preprocessor consists of different types of conditional compilation

<table>
  <tr>
    <td>#if</td>
    <td>Opening `if` condition</td>
  </tr>
  <tr>
    <td>#else</td>
    <td>`else` condition</td>
  </tr>
  <tr>
    <td>#elif</td>
    <td>`else if` condition</td>
  </tr>
  <tr>
    <td>#ifdef</td>
    <td>`if defined` condition</td>
  </tr>
  <tr>
    <td>#ifndef</td>
    <td>`if not defined` condition</td>
  </tr>
  <tr>
    <td>#endif</td>
    <td>`end if` condition</td>
  </tr>
</table>

Also, there are two alternatives for `#ifdef` and `#ifndef`, they are:

```cpp
#if defined(macro)
#if !defined(macro)
```

### Macro expansion


Macro's can also take parameters and replace them when called. For example:

```cpp
#define ADD(a,b) (a+b)

int main(int argc, char const *argv[]) {
	printf("%d\n", ADD(10,20));
	return 0;
}
```

### Line continuation

If you want to use complex macros you can use `line continuation` by add `\` at the end of the each line. For example:

```cpp
#define LOOPER(i) do \
                { \
                    printf("%d\n",i++); \
                } while (i < 3);
```

### Include guard

There might be a situation where you might have to define a header file in another header file and when called in a C++ file you might include both header files. When you compile this you will have a `Build fail`, to over come this we have to include something called as `Include guard`. It looks something like this

```cpp
#ifndef _HEADERNAME_H
#define _HEADERNAME_H
...
#endif
```

## Classes and Objects in C++

C++ is a an Object Oriented Program, that's what makes it different from C programming language. A class is define by using `class` keyword followed by class name. For example:

```cpp
class name_t {
	int i; // Data members
public: // Function members
	name_t (arguments); // Constructor
	~name_t (); // Destructor

};
```

Few points to remember:

* A class can have multiple constructor and only one destructor.
* A class when called and naming it is called an instance of that class. Example `name_t name;`, `name` is an instance of class `name_t`.
* Using classes you can allocate memory properly.

More information can be found [here](http://www.cplusplus.com/doc/tutorial/classes/).

### Defining Classes and Objects

There are different ways to define a class. For example

```cpp
class name_t {
	int i;
public:
	void some_name (arguments){ /* do something */};
};

int main(int argc, char const *argv[]) {
	name_t obj1;
	return 0;
}
```

Another way is to use `private` keyword, you can then use this to define `private` variables and function after `public`. For example:

```cpp
class name_t {
public:
	void some_name (arguments){/* do something */};
private:
	int i;
};

int main(int argc, char const *argv[]) {
	name_t obj1;
	return 0;
}
```

The public functions can be used outside, just declare it in the class file and define it outside the `class`. For example:

```cpp
class name_t {
public:
	void some_name (arguments);
private:
	int i;
};

void name_t::some_name (arguments){/* do something */};

int main(int argc, char const *argv[]) {
	name_t obj1;
	return 0;
}
```

The code can be divided into 3 stages:

* *Interface*: Usually kept in the header file.

	```cpp
	class name_t {
	private:
		/* data */
	public:
		void some_name (arguments);
	};
	```
* *Implementation*: Usually kept in an implementation file

	```cpp
	void name_t::some_name (arguments){/* do something */};
	```
* *Usage*: Usually kept in `cpp` file

	```cpp
	int main(int argc, char const *argv[]) {
		name_t obj1;
		return 0;
	}
	```

### Data members

In C and C++ we can find keyword called `struct`, when used in C++ it is an object. The different between a `struct` and `class` is that, `struct` by default has `public` data members, where as `class` has `private` data members, everything else is the same. For example:

```cpp
struct name_t {
	/* data */
};

```

is same as

```cpp
struct name_t {
public:
	/* data */
};
```

### Function members

You can define a same function with different signatures in C++.

### Constructors and Destructors

A constructor can be used to send in arguments while initializing a class. Destructors are used to clear the memory after the program ends, in C++ destructor are always called at the end of the program by default.

### Implicit and Explicit conversion

By Default C++ does implicit conversion. To make an explicit conversion we need to use `explicit` keyword for a constructor.

For example:

```cpp
class name_t {

public:
	explicit name_t (arguments);
	virtual ~name_t ();

};
```

### Namespaces

Namespace in C++ acts like a scope to a group of classes, functions etc... A Namespace can be created by using `namespace` keyword. For example:

```cpp
namespace name {

};
```

### Using `this`

An object in C++ can access its own pointer, to do so, `this` keyword is used. You can print out the address of a pointer by using

```cpp
printf("%p\n", this);
```

### Operator overload: Member function

Any function that belongs to a class is called a member function. Operator overload can be a part of member function.

### Operator overload: Non-member function

Any function that does not belong to a class is called a  non-member function.

### Conversion operator

You can use `+=` to concatenate a string with a rational number that belongs to a member function.

### Using new and delete

C++ allows you to allocate and delete memory for different data types using two keywords, `new` - To allocate memory and `delete` - To deallocate memory. For example:

```cpp
class name_t {
private:
	/* data */
public:
	name_t (arguments);
	virtual ~name_t ();

};

int main(int argc, char const *argv[]) {
	name_t *a = new name_t(); // to allocate memory
	delete a; // to deallocate memory
	return 0;
}
```

## File IO

In this section we will look at how to read and write files using `fstream`.;

### Reading Files

Reading a file in C++ can be done using `ifstream`, this data type has many functions associated to it but we want `open`, `good` and `close`. `open` opens a file from the memory, `good` checks if the state of stream is good or not and `close` closes the file after reading from it.

Writing File

Writing file can be done using `ofstring`, like `ifstring`, this data type provides the same functions - `open` and `close`. If a file already exists with that name, its over written, this can be changed using `ios::app` option that appends any string given to it.


## Data Structures

_A data structure is a group of data elements grouped together under one name._ A structure can have multiple data types grouped together to form a way of representation.

### Structs

It has a group of data types that can be called by name. It can be represented as:

```cpp
struct STRUCT_NAME {
    DATA_TYPE NAME;
};
```

You can access them as:

```cpp
STRUCT_NAME some_name;
some_name.NAME
```