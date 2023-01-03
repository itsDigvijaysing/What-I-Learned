# Salesforce Admin Notes

### Product & Services of Salesforce:

1. Sales
2. IOT
3. Marketing
4. Service Cloud
5. App
6. Community
7. Analytics
8. Health
9. Commerce
10. Others...


**Programming Technologies Salesforce:**

1. Lightning Component Framework : UI Development framework
2. Apex : Salesforce language for programming
3. Visualforce : Markup Language (.vfp)

**Automation Tools Salesforce:**

1. Process Builder (Active)
2. Workflow Rules
3. Flows

**Relationship types:**

1. Master-Detail Relationship : Like inheritance, if parent object deleted than child object will also delete (cascade delete).
2. Lookup Relationship : Same as MD but without cascade delete. Inherited record don't get affected even if parent data is changed.

---

### Info

**Multi-Tenant Architecture :** Single DB for Multiple Clients.  
**Meta-Data Architecture :** Store Pages Layout (Accounts, Leads, Contacts).

**App exchange** same as Playstore we can create App & Publish/Sale your app using app exchange.  
Standard App are inbuilt app of salesforce & Custom app which we create.  


> We store data on Objects (like table).  
> Fields are like columns with different datatype.  
> Records are like all info & Values of User (like Row).
  
* **Validation Rule**: if Validation rule contain error detection formula so if validation rule detect something than generate error.
* We can use 1 Profile to assign it to multiple Users.
* Roles we can use for hierarchy for profiles.
* Trigger.new : for New Data & Trigger.old : for Old Data.

> Profiles : User can access only the things which there profile allows.  
> Permission Set: We can assign access to things.  
> Workflow Rule : Automation Purpose & it get triggered if rule criteria met.

**To create Apex class, Triggers, Visualforce Page, components we can use :**

1. Developer Console
2. Notepad Editor
3. Inline Editor (Can see the edited output result)


