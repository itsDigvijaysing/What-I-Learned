# Salesforce Dev Notes

## Lightning Web Components [LWC]:



---

## Apex Language

We can use apex prog language to add custom logic to our application. like Complex Validation, Transactional logic, Complex business process, logic actions, etc. It's OOP Language & allows execution of flow & control statement.

---

## Trigger

Trigger is an Apex Code that executes before or after changes occurs to Salesforce records.  
These changes includes operations like Insert, Update, Delete, Merge, Upsert and Undelete.  

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

**Types of Context Variable:**

1. Trigger.new: For New Data for insert, update & undelete trigger.
2. Trigger.old: It returns old version of records for update & delete trigger.
3. Trigger.newMap: It returns a map of Ids to the new versions of records.
4. Trigger.oldMap: Map of Ids to the old version of records.
5. isInsert: Returns true if the trigger was fired due to an insert operation.
6. isUpdate: Returns true if the trigger was fired due to an update operation.
7. isDelete: Returns true if the trigger was fired due to a delete operation.
8. isUndelete: Returns true if the trigger is fired after a record is recovered from recycle bin.
9. isBefore: Returns true if the trigger was fired before any record was saved.
10. isAfter: Returns true if the trigger was fired after all records were saved.
11. Size: Returns the total number of records in a trigger invocation, both old and new.

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
2. **Visualforce Controller:** A Visualforce controller is a set of instructions that specify what happens when a user interacts with the components specified in associated Visualforce markup, such as when a user clicks a button or link. Controllers also provide access to the data that should be displayed in a page, and can modify component behaviour.

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