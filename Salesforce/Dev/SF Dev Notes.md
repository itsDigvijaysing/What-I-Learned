# Salesforce Dev Notes

## Aura vs LWC

- Aura: 
    1. Too much Code to write
    2. Rendertinf wasn't optimized
    3. Modern features were not available like modules, classes, premises
    4. ES5(2009) old JS version

- LWC:
    1. Shadow DOM
    2. Web Components
    3. Custom Elements
    4. template & slots
    5. Better Performance / Faster Development
    6. Compatibility across browser

Lightning web components is a new programming model for building Lightning Components. It uses core conceps of web standards.

> We can compose aura components from Lightning web components but not other way around.  

> Aura Components & LWC can coexist on same page. for admin & end user they both appear as lightning component

## Lightning Components

Now you can build Lightning components using two programming models: **Lightning Web Components**, and the original model, **Aura Components**. Lightning web components are custom HTML elements built using HTML and modern JavaScript. Lightning web components and Aura components can coexist and interoperate on a page. To admins and end users, they both appear as Lightning components.

> **Lightning Web Components** uses core Web Components standards and provides only what’s necessary to perform well in browsers supported by Salesforce. Because it’s built on code that runs natively in browsers, Lightning Web Components is lightweight and delivers exceptional performance. Most of the code you write is standard JavaScript and HTML.  
LWC supports Explicit DOM (Manual) & Automatic DOM (Automatic Changes).  

> **Aura components** are the self-contained and reusable units of an app. They represent a reusable section of the UI, and can range in granularity from a single line of text to an entire app. Events. Event-driven programming is used in many languages and frameworks, such as JavaScript and Java Swing. An Aura component is a combination of markup, JavaScript, and CSS.

**Aura and LWC** are two popular frameworks for building web applications. Aura is a framework for creating components that can be used in any web application. LWC is a framework for creating Lightning Web Components, which are components that are designed to work with the Lightning platform.

**Decorator**:  
The Lightning Web Components programming model has three decorators that add functionality to a property or function.

1. @api : To Expose Public Property.
2. @track : It observe changes in some interval & If Field changes then it rerenders & display new value. 
3. @wire : Used to make connection with method of Apex Controller & to read salesforce data.

**Lifecycle Hooks**:  
A lifecycle hook is a callback method triggered at a specific phase of a component instance’s lifecycle.

[Lifecycle Hooks Link](https://developer.salesforce.com/docs/component-library/documentation/en/lwc/reference_lifecycle_hooks)

- constructor()
- connectedCallback()
- renderedCallback()
- render()
- disconnectedCallback()
- errorCallback(error, stack)

---

## Apex Language

We can use apex prog language to add custom logic to our application. like Complex Validation, Transactional logic, Complex business process, logic actions, etc. It's OOP Language & allows execution of flow & control statement.

The Apex programming language is similar to one you probably already know and love—C#. Apex is saved, compiled, and executed directly on the Lightning Platform. Like C#, it’s object oriented.
  
![Invoking Apex](../Assets/Invoking%20Apex.png)
  
Apex offers multiple ways for running your Apex code **Synchronously** & **Asynchronously**.

> **Synchronous Apex :**  
> Synchronous term means existing or occurring at the same time. Synchronous Apex means entire Apex code is executed in one single go. In a Synchronous call, the thread will wait until it completes its tasks before proceeding to next. In a Synchronous call, the code runs in single thread.

> **Asynchronous Apex :**  
> Asynchronous term means not existing or occurring at the same time. Asynchronous apex is executed when resources are available. So any calling method which calls Asynchronous apex wont wait for outcome of Asynchronous call. Calling method will go on further execution in code. And Asynchronous execution happens in separate thread and then it will return to main program.  
> To Test makesure to enclose test code in startTest() & stopTest().  

**Types of Async Apex:**  

![Async Apex Types](../Assets/Async%20Apex.png)

1. **Future Methods** :  
    We use @Future Annotation & method should be static, as it only return Void type.  
    Can only accept Primitive Datatype & sObject not allowed in paramenter.  
    It can not call another Future Method & (Callout=True) to make API callout.  
    Max 50 Future Method per class.

2. **Batch Apex** :  
    To use it we need to write Batch Class & it should be global class/Method.  
    chunks is like Packet of Data. It can be scheduled through UI & It can also call Future Method.

    It include 3 main Methods :  

       1. Start (Return Chunk) :
            It Execute only Once.
            Collects the records or Objects to pass to the interface method (Execute).  
            Its First Method called when Batch Apex Runs.  
            It returns Database.QueryLocator Object that contains the record passed as chunks.   

       2. Execute (Void) :
            It Execute multiple times as per chunks amount.  
            Use it to do required processing/Action on each chunk of data.  
            We use Database.BatchableContext object & list.

       3. Finish (Void) :
            It Execute only once.  
            Called after all batches are processed by Execute.  
            Normally used to do post processing or general email confirmation after execution.

3. **Queueable Apex** :  
    In Queueable Apex We can chain Jobs & monitor using JobID, It also support sObject & Non-Primitive Data type.  
    It's Superset of Future Methods & It will work as combination of future & Batch Apex.  
    Max 50 Jobs can b eadded in queue for single transaction & Max Stack depth (nested) for chained Job is 5.
    To Submit Job we use System.enqueueJob method & it returns JobID.

4. **Scheduled Apex** :  
    Using Scheduled Apex we can run Apex class at Specific Time. e.g. Run Maintenance tasks on scheduled daily/weekly time. Scope should be Global class & Methods, We can schedule through Apex & Salesforce UI.  
    Time Sequence: Seconds_Min_Hours_DayofMonth_Month_DayofWeek_Year    
    It's Method: global void execute (SchedulableCotext ctx){}   
    For Scheduling: System.schedule('txt',vartime,instance class)   

**Wrapper Class**: is an object created in Apex code. This data lives only in code during your transaction and does not consume database storage. It’s a data structure or an abstract data type that can contain different objects or collections of objects as its members. We can store any different datatype or sObject inside Wrapper. It act like container to store data.

**Static Method**: Method that belongs to class & can be called without creating instance.  

---

## Trigger

Trigger is an Apex Code that executes before or after changes occurs to Salesforce records.  
These changes includes operations like Insert, Update, Delete, Merge, Upsert and Undelete.  
Callout Process should be made Asynchronously from trigger so trigger process will not be blocked.  

**Trigger Order of Execution:**  
Before Trigger -> System Validation Rule -> Record Saved -> After Trigger -> Assignment Rules -> Auto Response Rules -> Workflow Rules... 

**TYPES OF APEX TRIGGERS:**
1. **Before Triggers:** These are used to update/modify or validate records before they are saved to database.
2. **After Triggers:** These are used to access fields values that are set by the system like recordId, lastModifiedDate field. Also, we can make changes to other records in the after trigger but not on the record which initiated/triggered the execution of after trigger because the records that fire the after trigger are read-only.


**Syntax & Trigger Events:**
```java
trigger TriggerName on ObjectName (trigger_events)
{
    //code
}
```

Where **trigger-events** can be a comma-separated list of one or more of the following events:

- Before insert
- Before update
- Before delete
- After insert
- After update
- After delete
- After undelete
- merge,upsert

**Types of Context Variable:**

1. **Trigger.new**: For New Data for insert, update & undelete trigger.
2. **Trigger.old**: It returns old version of records for update & delete trigger.
3. **Trigger.newMap**: It returns a map of Ids to the new versions of records.
4. **Trigger.oldMap**: Map of Ids to the old version of records.
5. **Trigger.isExecuting**: True if trigger Context Executing.
6. **Trigger.isInsert**: Returns true if the trigger was fired due to an insert operation.
7. **Trigger.isUpdate**: Returns true if the trigger was fired due to an update operation.
8. **Trigger.isDelete**: Returns true if the trigger was fired due to a delete operation.
9. **Trigger.isUndelete**: Returns true if the trigger is fired after a record is recovered from recycle bin.
10. **Trigger.isBefore**: Returns true if the trigger was fired before any record was saved.
11. **Trigger.isAfter**: Returns true if the trigger was fired after all records were saved.
12. **Trigger.size**: Returns the total number of records in a trigger invocation, both old and new.

```java
trigger ApexTrigger on Opportunity (before update) 
{ 
 for(Opportunity oldOpp: Trigger.old)
 { 
 for(Opportunity newOpp: Trigger.new)
 {
 if(oldOpp.id == newOpp.id && oldOpp.Amount != newOpp.Amount)
 newOpp.Amount.addError('Amount cannot be changed'); // Trigger Exception
 }
 } 
}
```

---

## Visualforce

Visualforce is a framework that allows developers to build sophisticated, custom user interfaces that can be hosted natively on the Lightning platform. The Visualforce framework includes a tag-based markup language, similar to HTML, and a set of server-side “standard controllers” that make basic database operations, such as queries and saves, very simple to perform.  

1. **Visualforce Markup:** Visualforce markup consists of Visualforce tags, HTML, JavaScript, or any other Web-enabled code embedded within a single <apex:page> tag. The markup defines the user interface components that should be included on the page, and the way they should appear.
2. **Visualforce Controller:** A Visualforce controller is a set of instructions that specify what happens when a user interacts with the components specified in associated Visualforce markup, such as when a user clicks a button or link. Controllers also provide access to the data that should be displayed on a page, and can modify component behaviour.

A developer can either use a **standard controller** provided by the Lightning platform, or add **custom controller** logic with a class written in Apex.

> **Standard controller** consists of the same functionality and logic that is used for a standard Salesforce page. For example, if you use the standard Accounts controller, clicking a Save button in a Visualforce page results in the same behavior as clicking Save on a standard Account edit page.  
>If user doesn't have access to the object, the page will display an insufficient privileges error message. You can avoid this by checking the user's accessibility for an object and displaying components appropriately.

> **Custom controller** is a class written in Apex that implements all of a page's logic, without benefits of standard controller. If you use a custom controller, you can define new navigation elements or behaviors, but you must also reimplement any functionality that was already provided in a standard controller.  
> Like other Apex classes, custom controllers execute entirely in system mode, in which the object and field-level permissions of the current user are ignored. You can specify whether a user can execute methods in a custom controller based on the user's profile.

---

## Governor Limit

**Governor limits** are runtime limits enforced by the Apex runtime engine to ensure that code does not throw error.  
As Apex runs in a shared, multi tenant environment, the Apex runtime engine strictly enforces a number of limits to ensure that code does **not monopolize** shared resources. Resources such as CPU, processor, and bandwidth are shared by Apex on the Salesforce server.

[Governor Limit Sync/Async Link](https://developer.salesforce.com/docs/atlas.en-us.salesforce_app_limits_cheatsheet.meta/salesforce_app_limits_cheatsheet/salesforce_app_limits_platform_apexgov.htm)

Governor Limits:  
1. Monitor and manage platform resources such as memory, database resources, etc.
2. Enable multi tenancy by ensuring the resources are available for all tenants on the platform.
3. Can issue program-terminating runtime exceptions when limit is exceeded.
   
Governor Limits basically cover:
1. Memory.
2. Database Resources.
3. Number of script statements.
4. Number of records processed

---

## SOQL [Salesforce Object Query Lang]:

Salesforce Object Query Language. It is used to retrieve data from Salesforce database according to the specified conditions and objects. Similar to all Apex code, SOQL is also case insensitive.

Use SOQL when you know which objects the data resides in, and you want to:
- Retrieve data from a single object or from multiple objects that are related to one another.
- Count the number of records which meet the specified criteria.
- Sort results as part of the query.
- Retrieve data from number, date, or checkbox fields.

A SOQL query is equivalent to a SELECT SQL statement and searches the org database.  
Example: 
``` sql
Select name from account // standard object
Select name, Student_name from student__c // custom object
```

---

## SOSL [Salesforce Object Search Lang]:

Salesforce Object Search Language SOSL is a highly optimized way of searching records in Salesforce across multiple Objects that are specified in the query. A SOSL query returns a list of list of sObjects and it can be performed on multiple objects.  
The basic difference between SOQL and SOSL is that SOSL allows you to query on multiple objects simultaneously whereas in SOQL we can only query a single object at a time.

Example:
1. Returning: If we want to return texts from a particular object then we use returning keyword.  
    ``` sql
    List<List<sObject>> results = [FIND 'Univ*' IN NAME FIELDS RETURNING Account, 
    Contact];
    List<Account> accounts = (List<Account>)results[0];
    List<Contact> contacts = (List<Contact>)results[1];
    ```

2. Return specified fields: If we want to search text in a particular field, we are going to use search group.  

    Find {contact} IN (searchgroup)  
    ➔ All Fields (By Default)  
    ➔ Name Fields  
    ➔ Email Fields  
    ➔ Phone Fields  
    Find {contact} IN (searchgroup) returning objects & fields.  
    ``` sql
    List<List<sObject>> results = [FIND 'Univ*' IN NAME FIELDS RETURNING
    Account(Name, BillingCountry), Contact(FirstName, LastName)];
    List<Account> accounts = (List<Account>)results[0];
    system.debug(accounts[0].Name);
    ```
---

## sObjects and Generic sObjects:

**sObject:**  
Apex is tightly integrated with the 
database, we do not have to create any database connection to access the records or insert new 
records.  
Instead, in Apex, we have **sObjects** which represent a record in Salesforce.  

For example:
```sql
Account acc = new Account(Name=’Disney’);
```
The API object name becomes the data type of the sObject variable in Apex.  

- Account = sObject datatype
- acc = sObject variable
- new = keyword to create new sObject Instance
- Account() = Constructor which creates an sObject instance
- Name = ‘Disney’ = Initializes the value of the Name field in account sObject


**Generic sObject:**  
Generally while programming we use specific sObject type when we are sure of the instance of the sObject but whenever there comes a situation when we can get instance of any sObject 
type, we use generic sObject.

Generic sObject datatype is used to declare the variables which can store any type of sObject instance.

For example:  
sObject Variable ---> Any sObject Datatype instance
```sql
sObject s1 = new Account(Name = ‘DIsney’);
sObject s2 = new Contact(lastName = ‘Sharma’);
sObject s3 = new Student__c(Name = ‘Arnold’)
```

---

## Local Properties & Data Binding

- In Lightning component each component is class.
- Class contains properties and methods.
- Properties are variable to store data can be of type (undefined, number, string, objects, boolean, array/list).
- The properties are avaiable inside class only & to which you can't access outside the class are called Local properties.

**Data Binding** - In LWC is the synchronization between the controller JS and the template (HTML). basically when we map data from backend (js)(like server side apex) to frontend (html) that's called data binding.

```
<!-- One Way Data Binding -->

fullname = "Champ"  --> JS (Controller)
My Full Name is {fullname} --> Template (HTML)
```

- In template we can access property value directly if it's primitive or object.
- DOT notation is used to access the peroperty from an object
- LWC doesn't allow computed expressions like Names[2] or {2+2}.
- The property in {} must be a valid JS identifier or memver expressions. Like {name} or {user.name}
- Avoid adding spaces around the property, for example {data}
- Two way data binding also present. 

---

## Component Composition

Composition is adding component within the body of another component.
Composition enables you to build complex component from simpler building-block components.

- Advantages :
    - Composing, applications, and components from a collection of smaller components make code reusable and maintainable.
    - Reduces the code size and improves code readability

[Details & Example](https://developer.salesforce.com/docs/component-library/documentation/en/lwc/create_components_compose_intro)

---

## Debug

The Apex Replay Debugger can be used with all unmanaged code in all orgs. It works with Apex classes, triggers, anonymous Apex, and log files. This is an easy-to-use debugger that fits the majority of debugging use cases  .

Apex Replay Debugger is a free tool that allows you to debug your Apex code by inspecting debug logs using Visual Studio Code as the client. Running the replay debugger gives you the same features you expect from other debuggers. You can view variables, set breakpoints, and hover over variables to see their current value. 

Checkpoints are similar to breakpoints in that they reveal a lot of detailed execution information about a line of code. They just don’t stop execution on that line.

---

## Salesforce Developer Experience (SFDX)

**Scratch Org** : It's source driven and disposable deployment of SF code and metadata. Scratch orgs are driven by source, sandboxes are copies of Production. They do not replace sandboxes.

**Dev Hub** : It is the main salesforce org that you will use to create and manage your scracth orgs. (Like Developer Org)

---

## Salesforce Best Practices

- Create a code with mindset of using it for Bulk of data, so the code should be able to handle multiple record at once effectivery.
- Use Single Trigger per Object as in multiple trigger order of execution can not be determined.
- Rather than storing value in variable & than using it for loop we can directly assign SOQL query inside loop condition.
- Don't copy paste same code everywhere, we can just create it at one location and create reusable class & call it where we need.
- Avoid Nested loops & think about strategies to optimize for Time Complexity, Space Complexity & Limits.
- Avoid Business Logic in triggers, we can just write that logic somewhere & call it when we need to execute.
- Avoid Returning JSON to Lightning components because when we convert and store info it uses lot of heap memory. so, we might reach governor limit due to it. so, we can just send 'this object' directly & let platform handle it automatically for us.
- Good Naming to methods & variables.
- Add detailed Commit when something is sophisticated in code.
- Create Ticket everytime when i modify something & add it's details in the log.
- In Test classes for sample data use TestSetup Annotation Method & always test Batch of data for cases.
- Always Write Code in Alignement & Remember to take backup of your data everyday in cognizant onedrive. **<-Important**
- While working make sure you are working on updated code because if your code is old one then when you make changes & update it, then it will overwrite new code.
- There are lots of limits to be aware of, and they tend to change with each major release. Additionally, it’s not uncommon for limits to get looser rather than tighter, so be sure to check out the latest by looking at the Execution Governors and Limits link in Resources.
---