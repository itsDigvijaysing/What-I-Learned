# JavaScript Course Udemy

## Notes

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