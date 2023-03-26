# JavaScript Course Udemy

## Notes

- History of JavaScript  
  ![History of JS](Assets/History%20of%20JS.png)
- During Production: to make your app Compatible to all Browser, developer convert code to ES5 or ES6 JS version using BABEL.
- JS is High-Level, Object-Oriented, Mult-Paradigm Programming Language.
- JS is High Level lang because we don't need to worry about complex stuff like memory management.
- JS is Multi Paradigm means we can use different styles of programming.
- JS is also Object oriented means it's based on objects, for storing most kinds of data.
- Role of JS in Web Development:
  - HTML - Contect of Page (Frontend)
  - CSS - Presentation & styling
  - JS - Programming Lang for Dynamics & Interactive Tasks (Backend)
- React, Angular, VueJS this framework is based on JS.
- In Browser console we can do all basic JS so use it for basic tasks.
- We can use JS on web browser (Frontend Apps) or we can use Web Server (Backend Apps) like NodeJS to run JS without browser.
- React Native, Ionic apps are also using JS & using them we can build Native or Hybrid Apps.
- JS is also updating as per their new releases (updates) ECMAScript (ES Update) (Yearly Update).
- We can write Script tag & either in body, head or separate because it work as inline code but when we define file.js & we want to link it then we link it in head (good Practice) same way as we link css file.
- We should use camelCasing(start with lowercase letter) in JS even when naming variables & If something is constant or fixed then we start that variable name with upper case letter.
- We can not use reserved words or some symbols or starting with no. while naming.
- Value can be either Object or Primitive.
  ```js
  // Object Value
  let me = {
    name: 'Champ'
  };

  // Primitive Value
  let firstName = 'Champ';
  let age = 30;
  ```

- JS has dynamic typing so we do not need to define the data type of the value stored in variable as it is determined automatically as per value.
- JS check letter casing (case sensitive) as well so make sure to have exact variable name.
- mutate (mutable) meaning changed or reassign the variable value.
- **const** variable is immutable variable so we need to assign value when defining const.
- **let** are only available inside the block where they're defined. **var** are available throughout the function in which they're declared.
- We can write variable without declaring let, var, const type by directly giving it name & value but it's very bad practise.
  ```js
  //It's bad practise as it create it on property on global level.
  withoutVariableType = 'Hello World';
  console.log(withoutVariableType);
  ```
- Assignment Operators:  
  ```x=y, x+=y, x-=y, x++, x--```
- Comparison Operators:  
  ```>, <, <=, >=```
- Assignment Operator, Exponential are right-to-left & other mathematical operator are left-to-right direction. [Full Operator Precedence](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Operator_Precedence)
- Template Literals are helpful if we want to insert multiple variables & **expressions** (Not Statement) in the line of string.
  ```js 
  // We write Template Literals in `` & it's easy way to write long string with expressions.
  const name = 'Digvijaysing';
  const myName = `My Name is ${name} & I'm ${80-57} years old Engineer!`;
  console.log(myName);
  // Multiple Lines Using Template Literals
  console.log(`String with 
  Multiple 
  Lines`);
  ```
- We can also use Multiple Lines using ```\n\``` in string.
- Ternary Conditional Operator are used rather than if else statement to do simple conditional Operations & it produces value as expressions so we can also use them at place of expressions.
  ```js
  // tip is using ternary operator method & giving tip based on condition of bill amount.
  const bill = 275;
  // 50<=bill<=300 will not give correct output.
  let tip = (50<=bill && bill<=300 ? bill*15/100 : bill*20/100); //Ternary Conditional Operator
  console.log(`The bill was ${bill}, the tip was ${tip}, and the total value ${bill+tip}`);

  ```
- **String + No** = It will concatenate as string but **String - No** = it will perform as normal operation.
- Type Coercion refers to the process of automatic or implicit conversion of values from one data type to another. This includes conversion from Number to String, String to Number, Boolean to Number etc
  ```js
  console.log('9'-'5'); // Output: 4
  console.log('19'-'13'+'17'); // Output: 617 because Int + String means Concatenation.
  console.log('19'-'13'+17); // Output: 23 because '19'-'13' get converted to int
  console.log('123'<57); // Output: false
  console.log(5+6+'4'+9-4-2); // Output: 1143
  ```
- NaN = Not a Number, e.g. when we convert string with alphabet to Number.
- We can convert type to Number(),String(),Boolean(). 
- falsy values : 0, '', undefined, null, NaN.
- truthly value : any string or non falsy value.
  ```js
  // We can use falsy values in coditional statement
  // We cam also use it to check if variable defined or not
  let money = 0;
  //console.log(Boolean(money));
  if(money){
      console.log('Wow, You Have Money');
  }else{
      console.log('You should Earn Money');
  }
  ```
- Strict Equality Operator (Coercion Blocked): ```===``` (Type also checked)  
  Loose Equality Operator (Coercion Happen): ```==``` (Type not checked)
- Easy way to take input from user through Browser ```let inputValue = prompt("String");```
- Switch Conditional Statement in JS & we can also convert switch to multiple if else.  
  ```js
  switch (key) {
    case value1: //Here case will be checked key === value1 (Strict Equality Operator)
    case value2:
      //thing for both value1,value2
      break;

    case value3:
      // thing for value 3
      break;

    default:
      //thing if no case satisfy
      break;
  }
  ```
- We use AND by using **&&**, OR by using **||**, Not by using **!**.
- Expression: Any unit of code that can be evaluated (produce) to a value is an expression. Since expressions produce values, they can appear anywhere in a program where JavaScript expects a value such as the arguments. 
  ```
  Expressions
  1 → produces 1
  "hello" → produces "hello"
  5 * 10 → produces 50
  num > 100 → produces either true or false
  isHappy ? "🙂" : "🙁" → produces an emoji
  [1, 2, 3].pop() → produces the number 3
  ```
- Statement & Expressions of JS in [Details](https://www.joshwcomeau.com/javascript/statements-vs-expressions/)
- Statement: A statement is an instruction to perform a specific action. We use expressions in statement to perform action, Such actions include creating a variable or a function, looping through an array of elements, evaluating code based on a specific condition etc. JavaScript programs are actually a sequence of statements.\
  ```
  Statements
  let hi = 5;

  // if-else statements
  if (hi > 10) {
  // More statements here
  }else{
    //More Statement
  }

  throw new Error('Something exploded!');
  ```

- typeof undefined is undefined
- typeof null is object

- undefined == null //true
- undefined === null //false

- **Var** : When you declare a var variable it automatically assigns to window object  
  ```js
  var course = "Zero to Hero"; //global scope
  console.log(window.course); //Zero to Hero
  ```

- Spread Operator (...)  
  The operators shape is three consecurtive dots and is written as -> ...  
  Usage: 
  1. Expanding String: convert string into list of array
  2. Combining Array: Combine array or add value to array
  3. Combining Object: Combine object or add value to Object
  4. Creating new shallow copy of arrays and objects

  e.g.  
  ```js
  let greeting = "Hello";
  let charlist = [...greeting];
  console.log(charlist); // "H","E","L","L","O"
  ```

- Destructuring : The two most used data structure in JS are object and array.  
  Destructuring is a special syntax that allows us to "unpack" arrays or objects into a bunch of varuables, as sometimes that's more convenient.
  1. Array Destructuring:
  ```js
  let arr = ["amazon","google"];
  let [company1,company2] = arr;
  console.log(company1); // amazon
  console.log(company2); // google
  ```

  2. Object Destructuring
  ```js
  let options = { 	
        title: "Zero to Hero", type: "CRM"
      }
  let {title,type} = options;
  console.log(title) // Zero to Hero
  console.log(type) // CRM
  ```
  - Note : If we want varuable name as same name as property then go with destructuring

## Object / JSON Opertations
1. Object.Keys() = returns the key/property of obj
2. Object.Values() = returns the values of obj
3. JSON.Stringify = Converts the obj into string
4. JSON.Parse = converts the string into obj

## Array Method 

- map() : loop over the array & return new array based on given range, value return
- every() : true if every element of array satisfy the codition
- filter() : return new array with only the values which satify condition
- some() : true if at least one element of array satisfy the condition
- sort() : sort the elements of array
- reduce() : Reduces the array to single values left to right
- forEach() : call for each element of array

```js
array.methodName(Function(currentItem, Index, actualArray){
  //check example online
})
```

## Query Selector

Queryselector() method returns the first element that matches a specific css selector in the document.  

To return all elements using selector use QuerySelectorAll() method.  
```js
document.queryselector(selector);
```


## Events

### HTML Event Handler Attribute :  
When we add event through HTML, event always begin with on keyword like onclick, onchange, onkeyup, etc.

### Event Listener :  
Provides two methods for registering & deregisterin event listener.

  1. addEventListener()
  2. removeEventListener()

## Event Propagation

Event Propagation determines in which order the elements receive the event. There are two ways to handle this event propagation order of HTML DOM is Event Bubbling and Event Capturing.

1. Bubbling (Bottom to top) : When an event happens on a component, it first runs the event handler on it, then on its parent component, then all the way up on other ancestors’ components. By default, all event handles through this order from center component event to outermost component event.
   
2. Capturing (Top to Bottom) : It is the opposite of bubbling. The event handler is first on its parent component and then on the component where it was actually wanted to fire that event handler. In short, it means that the event is first captured by the outermost element and propagated to the inner elements. It is also called trickle down.

Custom Events are also present in JavaScript.

## Arrow Function

Normal Code  
```js
hello = function() {
  return "Hello World!";
}
```

With Arrow Function
```js
// Arrow Functions Return Value by Default.
hello = () => {
  return "Hello World!";
}
```

### this keyword

In regular functions the **this** keyword represented the object that called the function, which could be the window, the document, a button or whatever.

With arrow functions the **this** keyword always represents the object that defined the arrow function.  
