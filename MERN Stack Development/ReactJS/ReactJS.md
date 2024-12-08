- #ReactJS is [JavaScript](../../JavaScript/JavaScript.md) library 
- React Learning Site [React](https://react.dev/learn)
- It's all about User Interface (highly reactive UI)
- React apps are made out of _components_. A component is a piece of the UI (user interface) that has its own logic and appearance. A component can be as small as a button, or as large as an entire page. React components are JavaScript functions that return markup.
```jsx
function MyButton() {  
return (  
<button>I'm a button</button>  
);  
}
```
- Now that you’ve declared `MyButton`, you can nest it into another component:

```jsx
export default function MyApp() {
  return (
    <div>
      <h1>Welcome to my app</h1>
      <MyButton />
    </div>
  );
}
```
- Notice that `<MyButton />` starts with a capital letter. That’s how you know it’s a React component. React component names must always start with a capital letter, while HTML tags must be lowercase.
- **JS** is simply a scripting language, adding functionality into your website, **JSX** allows us to write HTML elements in JavaScript and place them in the DOM without any createElement() and/or appendChild() methods. JSX converts HTML tags into react elements. You are not required to use JSX, but **JSX makes it easier to write React applications**.
- **JSX** is stricter than HTML. You have to close tags like `<br />`. Your component also can’t return multiple JSX tags. You have to wrap them into a shared parent, like a `<div>...</div>` or an empty `<>...</>` wrapper.
```jsx
function AboutPage() {
  return (
    <>
      <h1>About</h1>
      <p>Hello there.<br />How do you do?</p>
    </>
  );
}

```
- Hooks: Functions starting with `use` are called _Hooks_, Hooks are more restrictive than other functions. You can only call Hooks _at the top_ of your components (or other Hooks).
- The code in `App.js` creates a _component_. In React, a component is a piece of reusable code that represents a part of a user interface. Components are used to render, manage, and update the UI elements in your application
- When Importing something like Images/files from parent folders to src folder react have some issues. The `App.js` file is located in the `src/` directory, so the `../` prefix tries to import the `Example.js` file from outside the `src/` directory which causes the error. All of the files you import in a React.js project have to be located under the `src/` directory.
	1. Either we have to move files inside `src` folder or where the `code.js` present, so we assign value to variable & use it. `e.g. logo`
	2. Or we use Importing files from the public/ directory with absolute imports, to import below directory `cat.png` we can use `img src="/cat.png"` directly & it's allowed as per absolute import in react.
```shel
	bobbyhadz-react/
		  └──src/
		    └── App.js
		  └──public/
		    └── cat.png
```
