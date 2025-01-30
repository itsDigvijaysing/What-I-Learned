# #Python Notes & Sample Codes

## Table of Contents

* Variables
* Datatypes
* Strings
* Booleans & Conditions
* Loops
* Tuples & Sets
* Modules
* Iterators & Generators
* [Lists](Python%20Extra/lists.md)
* [Dictionaries](Python%20Extra/dictionary.md)
* [Functions](Python%20Extra/functions.md)
* [Lambdas](Python%20Extra/lambdas.md)
* [Debugging](Python%20Extra/debugging.md)


### Variable in Python 
In this section we are going to learn how to represent variables in the Python language. Please note that variables in Python are dynamically typed, which means unlike languages like Java and C++, you don't need to specify whether the variable is a string, integer, etc.

There are some basic conveniences Python variables provide:

* Variables can be reassigned at any time
* We can also assign variables together (more on this later)
* We can also assign variables to each other

```Python
x = 100
y = 10
all, us , about = 10, 11 ,12
# In this case we have assigned variables at once
print(x+y) # This will print 110
print(us) # To see if it is going to print the us variable that we declared
``` 


### Datatypes in Python

* Bool: True or False values
* int : It's an integer like 5,3,4
* str : It's a sequence of characters for example "Your name" this is a string
* list : It's an ordered sequence of different data types for example ['1','apple','10.5']
* tuple : It's an immutable ordered sequence of different data types: ('1', 'apple', '10.5')
* dict : It's a collection of key value pairs {'name':'john'}
* set : It's a collection of unique data: {1, 'apple', 10.5}
* frozenset : It's an immutable collection of unique data: frozenset({1, 'apple', 10.5})
```Python 
x = true # Boolean
y = "Hello World" # This is a string
numbers = [1, 2, 3, 'four'] # this is a list
letters = ('a', 'b', 'c', 'd') # this is a tuple
v = {
    'name':'john',
    'address':'xyz street'
} # Its a dictionary in python

my_set = {1, 'apple', 10.5} # this is a set
my_set.add('banana') # this adds 'banana' to my set : my_set = {1, 'apple', 10.5, 'banana'}
my_set.add('apple') # this won't add nothing because apple is already in the set
my_set.remove('apple') # this removes 'apple' from the set: my_set = {1, 10.5, 'banana'}

my_frozen_set = frozenset(my_set) # this freezes my_set
```

#### Important Note
* Python is dynamically typed language it means that variable can be changed readily but its different in languages like JAVA, C++ C or etc. where we have to specify the variable type that's why they are called statically typed languages
* **None Key Word in Python is just like Null**
* **We can use type() Function to see the data type**



### Strings in  Python
* Strings in python can be in either double quotes or single quotes
```Python
my_str = 'Hello World'
string = "Hello"
```

#### String Concatenation
* We Concatenate strings using **+** operator 
```Python
str1 = 'Hello'
str2 = 'World'
print(str1+str2)
#This will print out Hello World
```
* You cannot concatenate strings with integers

```Python
8 + 'hello'
# This will give an error
but we can do this using an str() function and it will convert the the int into str then you can concatenate
```

* You can also use the **+=** operator to concatenate strings

```Python
string = "Cat"
string += " Dogs"
print(string)
#This will print out Cat Dogs
```

#### Formatting Strings
There's a new method to interpolate the strings 
* The method is called the F-strings this will help you convert the int value into a string 
```Python
x = 10
print(f"I have told you  {x}  times to clean the floor correctly")
# This will print I have told you 10 times to clean the floor correctly
# This a new way of interpolating string 
y = 'I have told you {x} times to clean the floor correctly'.format(x)
# This an old way of doing it and its applicable in **python 2.7**
```

#### String Index
```Python3
x = "Hello"
print(x[0])
#This will print out H 
# [] We use this for index
# This is useful for lists
```

#### Converting Datatypes

* We use built-in functions to convert the data types.
* For example int() str() float() 
* int () this is a built-in function that is used to convert a number into an integer.
* float() This is used to convert the number into a float
* str() This is used to convert to a string
**Note: We used input() function to get the user input**
```Python3
x = input('What is your age') #this will prompt to the user
print(x) # whatever user types it will be printed out
```


### Booleans and Conditionals

#### Condition

```Python3
# A psuedo code example
if some condition is true:
    do this
elif Some other condition is true:
    Do this
else:
    do something else

```

#### is VS ==
* is and == operators are not the same
* is operator compares two things and checks if they are stored at the same location in the memory
* == operator checks if both values are true

```Python3
a = [1,2,3]
b = [1,2,3]
a == b #returns true
a is b # returns false
```
**Indentation really matters in Python Language and (:) These colons help us indent the blocks**
**You can have multiple (elifs)**

### Loops in Python
* While loops and For Loops in Python

#### For Loops 
* They can be used to iterate the number of elements in the lists,
* They can also be used to iterate the strings
* range function is used to iterate through numbers
```Python3
for item in iterable_object:
    print(item)
# This is the syntax for loops in python
# Item represents the iterator's position
# You can call it anything you want
# printing numbers using for loops
for number in range(1,100):
    print(number)
# This will print numbers from 1 to 99
for letter in "Hello":
    print(letter)
# This for loop will print out the characters in the word hello
```

#### Ranges
* They are usually used with for loops
* range(7) if we give in one parameter this will only print out numbers 0 to 6 
* range(1,8) if we give in two parameters this will print out numbers 1 to 7 
* range(1,10,2) In this example the thir parameter is used to tell how many steps should be skipped and this will print odd numbers 
* range helps you generate the sequence of number its a python built in function

```Python3
x = input('How many times I have told you clean the room ')
y = int(x)
for time in range(y):
    print(f'{time}: Clean Your room')
# A simple example of loops

```

#### While Loops
* While loops are just like for loops
* While loops only run if the conditional statement is true otherwise it will stop
```Python3
while some_condition:
    #do this
# An example 
user_input = None
while user_input != 'please':
    print('Sorry Access Denied')
```
* We need to be careful with while loops since they will continue forever if the condition is not true **It will ruin the program**

```Python3
#Another While Loop Example
num = 0
while num < 10:
    num +=1
    print(num)
```


#### Importance of Break keyword in Python
* It gives us the ability to get out of the loop
```Python
while True:
    command = input('Type exit to get out of this')
    if command == 'exit':
        break


for x in range(1,10):
    if x == 3:
        break
```

### Tuples and Sets
* Tuples are different from lists they use parenthesis
* They are immmutable and they can't be change or you can't delete a value from it.

```Python
x = (1,2,3,4,5)
#Its a tuple
```

#### Why use a Tuple
* They are faster than a list.
* They are safer less bugs and problems
* You can also use tuples as the keys in dictionary

#### Different Methods on Tuples
* .count() to see the repetitive element in the list
* .index()

#### Sets in Python
* There is no order and they don't have duplicate values
* You can't use index to access the items because there is no order

```Python
set = {1,2,3,4}
# This is how a set looks like
a = set{(1,2,3)}
```


#### Set Methods
* .add() it is used to add something to a set
* .remove() it is used to remove something from the set
* .discard() It is also used remove the element from the set
* .clear() it removes everything from the set

#### Mathematical Methods in Sets
* | This symbol represents union
* & This represents intersection

```Python
a = {1,2,3,4}
b = {2,3,5}
a | b #This will print {1,2,3,4,5}
a & b # This will print {2,3}
```


#### Set Comprehension
* basically its like every other comprehension we have seen
```Python
x = {1,2,3,4}
b = {num+2 for num in x}
```

### Modules

Python modules allows us to reuse our own code or sometimes use somebody else's code.
We can write our own modules or we can use modules written by someone else like `requests`,`datetime` and `etc`.

**Note**: It's just a `python` file.

#### Built-in Modules
There built-in python modules that come by default.
List can be found here [LIST OF MODULES](https://docs.python.org/3/py-modindex.html)

- We can import modules by using `import` keyword
- We can also give modules an `alias` when we have long module names like `import random as r`.
- We can also import few functions from the modules `from random import randint`.
- If you want to import everything from random we do something like this `from random import *`

#### Custom Modules
Custom module is just file with python code.

##### For Example

```python
# file1.py
def hello():
    return "Hello"
def hey(name):
    return f'Hey {name}'
```

```python
# Importing a custom module
import file1 as fn
fn.hello()
fn.hey('Jake')
```

#### External Modules
External modules can be found here [PyPi](https://pypi.org/)
- We can install modules using `pip`, `pip install <package-name>`.
- Pip comes default in `Python 3.4` we can run using `python3 -m pip install <package-name>`.
- `print(dir(<package_name>))` This tells us about the attributes.
- `print(help(package_name))` This tells us everything about the package

#### Using PEP8 to cleanup Code

- We can use `autopep8` to fix whitespaces and ident our code
- We can use `autopep8 --in-place <file_name>`

#### The `__name__` variable
- The `__name__` variable usually refers to the file name if its the file then it will interpreted as `__main__` 
- If it's a module then `__name__` will be interpreted as the `__file_name__`.

**Note** : use `if __name__ = '__main__' `

### Iterators and Generators

#### Iterator
Iterator is an object that can be iterated upon. An object which returns data, one element at a time when next() is called on it.

#### Iterable
An object which will return an Iterator when `iter()` is called on it.

For example `"HELLO"` is an iterable but its not an iterator but `iter("HELLO")` returns an iterator.

#### `NEXT()`
When `next()` is called on an iterator, the iterator returns the next item. It keeps doing so until it raises a `StopIteration` error.

```python
def custom_iter(iterable):
    iterator = iter(iterable)
    while True:
        try:
            print(next(interator))
        except StopIteration:
            print("END")
            break
```

#### Generators
- Generators are iterators.
- Generator can be created with generator functions

### Python Extra Things

* [Lists](Python%20Extra/lists.md)
* [Dictionaries](Python%20Extra/dictionary.md)
* [Functions](Python%20Extra/functions.md)
* [Lambdas](Python%20Extra/lambdas.md)
* [Debugging

---
## Python Code Samples
### While Loop
- While can be used to start a script on the core part of it or to use as a logic operator.

```python
while True:
    answer = input("Do you want to try again? (Y/n) ")
    if answer == 'y' or answer == 'Y' or sys.stdin.isatty():
        num = int(input("Enter a number: "))
        print(num2roman(num))
    elif answer == 'n':
        break
    else:
        print("Invalid input.")
```

### For Loop
- For loops are essential these days, there is no doubt about that. It's possible to make a huge range of logical solutions with lists, arrays and other variables.

```python
for link in linksFinder:
    print(link.get("href"))

    saveFile = input(
        "Do you want to save this list inside a text file? (y/n) ")
    if saveFile == "y":
        with open("links.txt", "a") as file:
            file.write(link.get("href") + "\n")
    else:
        pass
```


### Random List
- Randomization is a must need thing in our code, so this is a quick snippet to random lists.

```python
import random

sample = ['a', 'b', 'c', 'd', 'e']
print(random.choice(sample))

# For cryptographically secure random choices (e.g., for generating a passphrase from a wordlist), use secrets.choice():

import secrets

sample = ['battery', 'correct', 'horse', 'staple']
print(secrets.choice(sample))

```

### Random Dictionary
- Randomization is a must need thing in our code, so this is a quick snippet to randomize dictionaries.

```python
import random

visualSet = {"Rock": rock, "Paper": paper, "Scissors": scissors}
aiChoice = random.choice(list(visualSet.values()))

"""All possible ways to randomize:

'd' is the dictionary variable.

A random key:
random.choice(list(d.keys()))

A random value:
random.choice(list(d.values()))

A random key and value:
random.choice(list(d.items()))
"""

```
### Bytes Encode and Decode
- Bytes converts an object to an immutable byte-represented object of given size and data, which is useful for writing or reading HEX values inside a file.

```python
    # Bytes Encode and Decode Study

    def writeString():
        # Write a string at the end of a JPG file.
        with open(relative_to_assets('photo.jpg'), 'ab') as f:  # ab append bytes mode
            f.write(b' Hidden message: test :)') # b is for bytes

    def readString():
        # Read HEX of the JPG file.
        with open(relative_to_assets('photo.jpg'), 'rb') as f:  # Read bytes mode
            jpgContent = f.read()
            # when FF D9 occurs.
            offset = jpgContent.index(bytes.fromhex('FFD9'))
            f.seek(offset + 2)
            print(f.read())

    def deleteString():
        # delete everything after the last FF D9 from a JPG file
        with open(relative_to_assets('photo.jpg'), 'r+') as f:  # Read bytes mode
            jpgContent = f.read()
            offset = jpgContent.index(bytes.fromhex('FFD9'))
            f.seek(offset + 2)
            f.truncate()
```

### Endswith
- This function returns True if a string ends with the specified suffix (case-sensitive), otherwise returns False. A tuple of string elements can also be passed to check for multiple options. For startswith we have a similar approach.

```python
from pathlib import Path
import pathlib
files = os.listdir(ASSETS_PATH)

# Interesting solution to pick specific files inside a list.
for file in files:
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
      print(f"Optimizing {file}")
      imgOptimize = Image.open(relative_to_assets(str(file)))
      imgWidth, imgHeight = imgOptimize.size
    else:
      print(f"{file} is not a PNG or JPG, skipping")
```

### Regular Expression
- Regular expression is a special sequence of characters that helps you match or find other strings or sets of strings, using a specialized syntax held in a pattern. Regular expressions are widely used in UNIX world.

```python
import re
    
# Regular Expression from module re;
# https://docs.python.org/3/library/re.html
# validate def will make sure that the remTwo var can have a "." as float


def validate(string):
    result = re.match(r"(\+|\-)?\d+(\.\d+)?$", string)
    return result is not None
```
- Currently working on Project which uses Regular Expression to solve manual tasks for developer. [Github Repo SF CodeScan Fixer](https://github.com/DBRajput/SF_Security_Issue_Fixer.git)
### Input
- Input is a built-in function in Python, allows coders to receive information through the keyboard, which they can process in a Python program. This is basic and essential.

```python
# Simple like that :)
userString = input("Enter a text: ")
print(userString[::-1]) # Reverse the string
```

### Time Conversion
- Making mathematical operations inside python are easy things to do, you basically need to know the formula and the logic to implement conversion and precision operations.

```python
minutes = "24.785089"
print(minutes + " minutes is equal to " +
      str(float(minutes)/60/24/365) + " years.")
```

### Open Website or Application
- Opening external websites or applications also requires the OS module to be imported, this snippet allows the script to open almost every type of file, like Shell scripts, default applications and websites.

```python
import os
targetSite = "https://google.com/"
os.system(f"open {targetSite}")
```

### Simple API GET Method
- Reading values from an API can be done very easily by using the requests module and also to converting the API values into Python dictionaries using json module and function.

```python
import requests
import json

# Getting the API url/path
responseAPI = requests.get('https://randomfox.ca/floof')
# Output from GET: {'image': 'https://randomfox.ca/images/13.jpg', 'link': 'https://randomfox.ca/?i=13'}
generatedFoxImg = responseAPI.json()

print(f"Your random fox: {generatedFoxImg['image']} \n")
```


### FastAPI GET Method
- FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. Making GET requests are easy thing to do, just need to import the module and associate the function to a variable and start coding paths and parameters with the FastAPI Functions.

```python
from fastapi import FastAPI

appStudy = FastAPI()


@appStudy.get("/")
async def root():
    return {"messageField": "Message content here."}
```

### FastAPI POST Method
- The POST method is used to request that the origin server accept the entity attached in the request as a new subordinate of the resource identified by the Request-URI in the Request-Line.

```python
from fastapi import FastAPI, Path
from pydantic import BaseModel

appStudy = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    quantity: Optional[int] = None

# This is actually the API values
inventoryDict = {
    "1": {"name": "Bread", "price": 1.25, "quantity": "10"},
    "2": {"name": "Milk", "price": 2.45, "quantity": "5"},
    "3": {"name": "Eggs", "price": 3.99, "quantity": "20"},
    "4": {"name": "Cheese", "price": 4.99, "quantity": "15"},
    "5": {"name": "Butter", "price": 5.00, "quantity": "5"}
}

# Using POST method
@appStudy.post("/post-item/{item_id}")
def createItem(item_id: int, item: Item):
    # Let's create a new item id.
    if item_id in inventoryDict:
        return {"DataErr": "Item already exists"}
    else:
        inventoryDict[str(item_id)] = item
        return inventoryDict[str(item_id)]
```



### Key Obfuscation
- Obfuscation is the deliberate act of creating source or machine code that is difficult for humans to understand, this helps to improve security - it is far from being the ultimate security solution but is a thing to use in a non-production environment.

For creating a file with the key and obfuscating it:

```python
import base64
import pathlib
import os
import re
from pathlib import Path

# Dynamic File Path Solution
API_PATH = pathlib.Path(__file__).parent.absolute()


def relative_to_assets(path: str) -> Path:
    return API_PATH / Path(path)

userChange = input("Enter key: ").strip()

# Pick userChange and encode it to base64
userChange = base64.b64encode(userChange.encode('utf-8'))
# Save userChange to "API" file
with open(relative_to_assets('Data/security/API'), 'wb') as f:
    # Delete everything inside the file.
    f.truncate()
    f.write(userChange)

    print("DONE! You are ready to use the API!")

```

### XOR Cipher
- XOR Encryption is an encryption method used to encrypt data and is hard to crack by brute-force method, i.e generating random encryption keys to match with the correct one.

Encrypting:

```python
#! /usr/bin/env python3
import base64
import os
import pathlib
import re
from pathlib import Path

# Dynamic File Path Solution
KEY_PATH = pathlib.Path(__file__).parent.absolute()


def relative_to_assets(path: str) -> Path:
    return KEY_PATH / Path(path)


def encryptSecurity():
    # Use external script to make base64 or https://www.base64encode.org/
    key = "MTMy"  # up 255
    key = base64.b64decode(key)
    cleanKey = re.sub(
        r"[^A-Za-z0-9-]", "", key.decode("utf-8"))
    finalKey = int(cleanKey)

    loadEnc00 = open(relative_to_assets("Data/security/.KEY"), "rb")
    byteReaderData = loadEnc00.read()
    loadEnc00.close()

    byteReaderData = bytearray(byteReaderData)
    for index, value in enumerate(byteReaderData):
        byteReaderData[index] = value ^ finalKey

    Enc = open(relative_to_assets("Data/security/.KEY.nclmE"), "wb")
    Enc.write(byteReaderData)
    Enc.close()

    # Delete Data/security/KEY
    os.remove(relative_to_assets("Data/security/.KEY"))


encryptSecurity()

```

Decrypting:

```python
#! /usr/bin/env python3
import base64
import os
import pathlib
import re
import string
from pathlib import Path
import signal

# Dynamic File Path Solution
KEY_PATH = pathlib.Path(__file__).parent.absolute()


def relative_to_assets(path: str) -> Path:
    return KEY_PATH / Path(path)


def signal_handler(sig, frame):
    # If the program exits then remove important files.
    os.remove(relative_to_assets("Data/security/.tmp/.KEY"))
    exit()


def decryptSecurity():
    # Use external script to make base64 or https://www.base64encode.org/
    key = "MTMy"  # up 255
    key = base64.b64decode(key)
    cleanKey = re.sub(
        r"[^A-Za-z0-9-]", "", key.decode("utf-8"))
    finalKey = int(cleanKey)

    loadEnc00 = open(relative_to_assets(
        "Data/security/.KEY.nclmE"), "rb").read()

    byteReader = bytearray(loadEnc00)
    for index, value in enumerate(byteReader):
        byteReader[index] = value ^ finalKey

    decEnc = open(relative_to_assets("Data/security/.tmp/.KEY"), "wb")
    decEnc.write(byteReader)


try:
    # signal handler for "CTRL + C"
    signal.signal(signal.SIGINT, signal_handler)
    decryptSecurity()
    signal.pause()
except:
    # In exeption remove important files.
    os.remove(relative_to_assets("Data/security/.tmp/.KEY"))

```