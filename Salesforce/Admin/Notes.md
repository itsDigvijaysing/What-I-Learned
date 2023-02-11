# Salesforce Admin Notes

## Product & Services of Salesforce:

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

**Salesforce Edition** 

> ![Salesforce Edition](../Assets/Salesforce%20Editions.png)
> [Salesforce Feature Allocation per Edition link](https://help.salesforce.com/s/articleView?id=sf.overview_limits_general.htm&type=5&language=en_US)


**Relationship types:**

1. Master-Detail Relationship : Like inheritance, if parent object deleted than child object will also delete (cascade delete).
2. Lookup Relationship : Same as MD but without cascade delete. Inherited record don't get affected even if parent data is changed.

---

## Info

**Multi-Tenant Architecture :** Single DB for Multiple Clients.  
**Meta-Data Architecture :** Store Pages Layout (Accounts, Leads, Contacts).

**App exchange** same as Playstore we can create App & Publish/Sale your app using app exchange.  
Standard App are inbuilt app of salesforce & Custom app which we create.  

> We store data on Objects (like table).  
> Fields are like columns with different datatype.  
> Records are like all info & Values of User (like Row).
  
* **Validation Rule :** Validation rule contain error detection formula so if validation rule detect something than generate error.
* We can use 1 Profile to assign it to multiple Users.
* Roles we can use for hierarchy for profiles.
* Trigger.new : for New Data & Trigger.old : for Old Data.

**Storage Types :**
1. File Storage - Salesforce File & Custom Files.
2. Data Storage - Object, Data & Metadata. 

**To create Apex class, Triggers, Visualforce Page, components we can use :**

1. Developer Console
2. Notepad Editor
3. Inline Editor (Can see the edited output result)

---

## Access
> ![Access](../Assets/Access.png)

- **Org Access:** IP Range/Login hrs.
- **Object Access:** Control Profile Access/R/W permissions, Permission Set.
- **Record Access:** Role Hierarchy, Sharing Rule, Manual Sharing, OWD.
- **Field Access:** Field Level Control & Accessibility.

**Profiles:** User can access only the things which there profile allows. Profile are mandatory to access any object, field or other permission needed things. One User can have only one profile & through profile we can set all permissions.

**Permission Set:** They are used to give extra permission at User Level. They are not Mandatory. Multiple User can have same permission set & One User can have multiple permission set.

**Sharing Rules:** Use sharing rules to extend sharing access to users in public groups, roles, or territories. Sharing rules give particular users greater access by making automatic exceptions to your org-wide sharing settings.

**Sharing Model**  
> ![Security Model](../Assets/Security%20Model.png)

---

## Automation Tools Salesforce:
>[Flows, Workflow Rules, Process Builder Info Link](https://www.manras.com/salesforce-flow-vs-process-builders-and-workflow/)

1. Process Builder (Retiring)
2. Workflow Rules (Retiring)
3. Flows (Active)

**Interactive Experience Automation**:  
- Screen Flow
- AutoLaunched Flow
- Approval Process
- Lightning Components
- Visualforce Page

**Background Process Automation**:  
- Record Triggered Flow
- Schedule Triggered Flow
- Platform Event Triggered Flow
- Apex Code
  
## Salesforce Details:

- Company setting has all the information about the company and all licenses & things.
- [User Interface Setting](https://help.salesforce.com/s/articleView?language=en_US&id=sf.customize_ui_settings.htm&type=5) - Control features of how user interact with ui.
- If we set multi currency in salesforce then we can’t disable it later.
- My Domain helps in custom login page, custom login policy, single sign on & other third party login methods & also control policies, security stuff.
- App menu easy way to control visibility of app in salesforce
- App manager info about each app & greater customization controls, utility items, navigation items, branding changes, profiles access.
- In App Guidance - it allow us to guide user using different popup msg ways.
- List Views - list which we can customize using filters & fields to display, its available in 3 variations list view ,split view & kanban view.
- We can add multiple users using Add Multiple User (up to 10 users) or Data Loader (More than 10 users).
- In the user record overview we can do user related tasks such as unlocking users, freezing users and deactivating users. (User Delete not possible)
- User license (sets the baseline features as user can use), permission set license (for access to variety of tools & function) & feature license (for access of additional features).
- Delegated Administrator grants user admin access for some time (to share admin duties for some time)
- Login as any user (It can be helpful for admin to exp exactly what particular user is facing by login as that user) (Go to Login access policies & allow this option then login by going to users section as any user)
- Setup audit trail -  Used to check any salesforce metadata (any backend config) changes in our salesforce org (last changes to org (logs))
- Salesforce Path - its component in which we see the progress of current task and at the end we will see closed won or closed lost for opportunity.
- Actions - Global Action & Record Action (Object specific action) (easy way to do things (check google IMG))
- Opportunity splits - Used to split revenue credit for opportunities. (If multiple people (Team) working on same opp then after closed won it will split credit as set values (total of splits do not need to add up 100%))
- Quotes - They are basically like template on which we can add extra sales details. & opportunity can have multiple quotes but only 1 quote will synced to opportunity. (Quote Template, Quote PDF)
- Multiple things are by default disable so we have to enable it to use it.

## Security:

- Salesforce start with very restrictive access & we open up the access using various features
- Trusted IP Address - Allow users to login without any verification request or challenges.(In Network Access section)
- Login IP range, Login IP Hours can control access in org (In Profiles Section)
- Password Policies - Configure length, expiry date, complexity of passwords & many other things. Can be set at profile lvl or organization lvl.
- Identity Authentication / Device Activation - Allows to control when and how users are prompted to verify their identity.e.g.  Like allowing Multi Factor authentication. (In history we can also see how users verified their identity)
- Session Settings - Used to config users sessions settings such as time until timeout and session security.(In Identity verification section & its own)
- Security health check - rating of how good standard security in salesforce org with current config (we can also fix using fix button (there are different sections of config details))
- 3 layers of access -  object access, record access, field access.
- CRED Access - Create, Read, Edit, Delete.
- Object Access - Control what object user can see (In Profile section)(Tab Hidden- can’t see field, Default off - can search and access it, default on - present in navigation bar every time)
- By going in Standard Object Permission of profile we can control CRED access permission for fields.
- Record Access - Control which records users have access to and their level of access (OWD, Role Hierarchy, Sharing Rules, Manual Sharing, Team Access).
- Sharing Settings - Most restrictive baseline of access, It contains Sharing rules of 2 types owner based sharing rule & criteria based sharing rule.
- Manual Sharing - We can manually share records with anyone.
- Team Access - Enable Teams & it will allow to view the record of users of teams.
- Field Level Security / Access - Which field user can access. (In Profile Section (Read access & Edit access)) or we can also use Set Field-Level Security (Visibility / Read only) by going to the object then the field section.
- Org Wide Default - most restricted then by going in a sharing setting we can access the OWD access. It can not overrule over profile access. & record access can only be opened up & can’t be restricted.
- Roles & Roles Hierarchy - effective way for record access & through sharing settings we can enable or disable it for custom objects (not standard object). High level roles will have access to all low level role records. & it can’t overwrite / override profile settings (if profile setting does not give access to a specific object then it will not be visible even if the user is on a higher level in hierarchy then other users who have access.) & it will not overrule if users have specific CRED permissions.
- Manual sharing / Sharing Rules - (Sharing rule - max 300 Owner based or max 50 Criteria Based). (Manual sharing -  Record by record basis & give access with read only / edit ). In the sharing rule all records will be shared as per settings.
- Public groups - way of sharing records with groups of users who are not normally grouped together. We can add the roles, profiles, territory & records in profile groups and everyone in the profile group will have shared records.
- Teams - 3 types of teams- Account teams, opportunity teams, case teams. Go to setup and search teams. So basically we can add specific members in the account team for each account & give them access to account View/ Edit & their child cases/opportunities.
- Profile - One and only profile to each user. & standard profile can not be edited we can clone them and create custom ones. We can activate enhanced profile interfaces from User management settings. Very important for access management & permission sets can  be used to grant more access.
- Permission Sets - Used to grant more functionality to users than their profile permits. New permission set is very restricted then we open up as per our requirement.
- Permission set group - Its group of permission sets which we can assign to users rather than doing it manually for each permission set. We also have muted permission set so if we don’t want our assigned user to do some tasks of which was previously granted by permission set then we can create muted permission set which will mute the thing so assigned user will not be able to do it (removing permission granted by permission set group by using muted permission set).

## Objects:

- Standard Object- Can not be deleted but can be renamed or hidden. They normally come with standard fields.
- Custom Object - either from SpreadSheet or Standard. We can use a custom tab if we forget to create a tab at the time of the custom object.

### Opportunities:  
It's used to track and manage your potential deals with salesforce opportunities.

### Sales Process (Path for Opportunity):  
The Process Opportunity goes through…  
e.g. B2B, B2C, Tender (reflect in opp stage field)  
We can assign sales process by selecting it in record types of opportunities.  

### Forecases: 
Used to predict future sales revenue and such things…  
Forecast 5 type: omitted, pipeline, Best Case , Commit, closed.  

### Orders:
Used to Track fulfilment of product and services.  
To create order we need to have contract.  
- Reduction Orders - Return / Cancellation of Order
- Negative Quantity Settings - Negative amount of products.
- Zero Quantity - Order with Zero amount of products.
- others....

### Contracts:
Used to store information about a Contractual Agreement between Parties.
- Contract is Linked to Account (Required Field) & it also has its Status & Contract Term (Months Period) as required field.

### Products:
Used to represent Products or services sold.
- Can be linked to Opportunity, Quotes, Orders, etc.
- Product must have standard price set while accessing it with price book.
- It also has price book which store multiple products while linking.
- Price book determines at what price product will be sold at.
- When we set Products in Opportunity then it become Opportunity Line Item.

> Product Scheduling  
> Used for Payment and delivery cycles of products or services.  
> Quanitity Scheduling / Revenue Scheduling.

### PriceBooks:
Contains the Prices (Price book entries) that product should be sold at.
- 1 Standard PB & Multiple Custom Price Books.

## Fields:
- Field data type can be changed for custom field only not standard field. there are some restrictions
- we can only delete custom field not standard field.

# Salesforce Relations:

> Relations are used to link records or objects with each other.  
> Schema builder is a good way to know the related objects in UI..

**Master Detail Relation**:
- Tight relationship - 1 to 1, 1 to many, many to many using Junction object (fix cascade delete) it have rollup summary feature
- max 2 MD relationship per object
- we create relationships on child/Detail objects.
- In MD relationship detail objects inherit security & sharing from masters object.
- Master always required while creating records in details object in MD but not needed in Lookup relationship.

**Lookup Relationship**:
- Loose Relationship - 1 to 1,1 to many,Self,  External, Indirect, Hierarchical. (LR don't have rollup summary field).
- Max 40 in LR per object.
- Master is not required while creating child record.

**Extra Info**:
- Create relationships at child objects in parent child relationships.
- we can create lookup then convert it master detail

## Record types:

- It will show different picklist values / Page Layout as per the record type assigned.
- Show Picklist values, Page Layouts, Business Process as per Record Type assigned.
- Page layout per record type per profile.
- object can have multiple record types
- record type access set by profile.

- Record Type will check only while data creation but after data is created the user can access data of all record types layouts and can modify it as RT work while data creation after that it will not control the data.


## Business Process:
we can select which process picklist are needed.
used to specify which picklist values are available to users based on the record types for 
- Lead Processes - lead
- Sales Process - opportunity
- Support Process - case.

## Lightning Application:

[//]: # (This are comments not visible on preview)
[//]: <> (Work on it...)

- Lightning Page -  App Page (1 pg for salesforce lightning & Mobile App) , Home Page (Lightning Experience Home Page), Record Page (when we see record info)
