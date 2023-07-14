# Some Topics from FSC Workbook

- [FSC Videos Links](https://quip.com/hMviAKe9T10M)
- My Domain is required before you can use Lightning components.
- Check Advisor, Personal Banker and Relationship Manager Profiles.
- Person Accounts store information about people by combining certain account and contact fields into a single record. 
	- Usually, you see the Person Account as a single object, but sometimes she appears as both an account and a contact, such as in global search. Don’t worry—they’re always part of one person account and both parts appear when you click either one.
	- Person Account has its own standard object but it only contains page layout, Compact Layout, Record types & Flow triggers & nothing else (no fields & stuff) in this object (As all other things are present in Account & Contact Object)

- business account stores information in an account object and a contact object. The account object contains information about the business. The contact object contains information about the person you work with at that business.
- Relationship groups are custom object records used to store collections of accounts.  
    You can use relationship groups to manage the following:
    - A household of people who reside at the same address
    - An extended family consisting of multiple generations of relatives
    - A professional group such as a medical practice or a law firm
    - The trustees and beneficiaries of a trust

- Primary group : A client’s data can also be summarized with the data of others by making a client part of a group.When a group is a client’s Primary group, all the client’s financial information is usually rolled up into that group.
- Household: A household represents a group of clients and businesses whose financials are summarized at the
household level.
- In Financial Services Cloud client data like Tasks, Events, and Financial Accounts is aggregated or “rolled up” into the client’s financial summary to view summarized data. We are using the custom objects Rollup By Lookup Config and Rollup By Lookup Filter Criteria. These objects enable calculation of rollup summaries based on lookup field.
	- While we can’t customize rollup configurations, Financial Services Cloud provides several out-of-the-box rollup summaries at the person and group levels.
	-  can only set rollups to the primary group. If the Primary Group field is not enabled, the option to roll up is restricted.  
- Financial Services Cloud comes with Multiple Relationship Groups enabled. This lets advisors add a person to more than one relationship group. For data security reasons some clients may want to disable this feature.
- The Relationship Map and Group Builder can be used to manage visibility to the right level of Account & Contact data for the user. This is controlled by the users’ Profiles (e.g. Advisor, Banker, Teller, etc.).
- Reciprocal Roles: Groups or Households sometimes have relationships with people who don’t live in the same household. Financial Services Cloud provides several reciprocal roles, including lawyer/client, accountant/client, parent/dependent, business/proprietor, and more. If those roles don’t cover your business needs, it’s easy to add new roles.
- FSC has Life Events or Business Milestones Lightning component that can be displayed on the Account record page to create a more personal, need-based customer engagement by capturing and visualizing important life events, such as having a baby, changing jobs, or buying a home. It shows Life Events for a person account record page and business milestones for a business account record page.
	- If you add the component to a Person Account record page to show Life Events.
	- If you add the component to a Business Account record page to show business milestones.
	- Person Life Event Object New & it should be used (old - Life Event Object not being used).
- You can capture (Only 1 event) Once-in-a-Lifetime Events with selecting "Unique Event" Types to prevent users from creating more event of that event type.
- You can Hide Sensitive Life Event Types so it doesn’t appear on the component until you add a life event of that type for the person's Life Event Timeline.
- Life Events Details to show after Hover : When you hover over a life event, an expanded lookup card displays the key fields from the event record. You
can customize the associated compact layout. & We can also Personalize Life Events with Custom Icons (SVG image files).
- It's easy to create related Opportunities, Cases, Events or even Action Plans because of life Events & Business Milestones.
- financial goal can track a client’s progress toward major purchases, retirement savings, or other life goals & we can only create savings-oriented (You already have own money) goals. We can not create a goal for paying down a debt & also can’t associate a goal with a specific financial account. (We can associate it to household or maybe person account)
- Action Plans are provide quick & consistent way to do the task like onboarding Financial Accounts and resolving Cases. Action Plan Template defines the tasks (steps) needed for completing a repeatable business process. Each task is given a priority, assigned to a user and given a completion date. We can create action plans for tasks associated with Account, Campaign, Case, Contact, Contract, Lead, or Opportunity records & for tasks associated with custom objects with Activities enabled.
- Action Plan Template contains the steps of tasks and other items needed to complete a business process & each task given priority (No. of Days to Complete & who is responsible).
- Action Plans Templates: We can create Action Plan template using existing Template. We can make Different Versions of Actions plan Template.
- Action Plan Template Packaging: We can deploy our template to different organization with ease.(Test in Sandbox then Deploy in Production Org (Use Case))
- Action Plan Templates can now include document checklist items. An action plan template can have tasks, document checklist items or combination.
- We can create action plans for tasks associated with Account, Campaign, Case, Contact, Contract, Lead, or Opportunity records . 
- We can also create Action Plans for tasks associated with custom objects with Activities enabled.
- We have to assign the permission set licenses to the profiles of users that need access to Action Plans. 
- Intelligent Referrals: Intelligent need-based Referrals & scoring is referral management workflow that helps check & select referrals in business.
  - User create & automatically route referrals based on customners expressed Interest.
  - Dashboard & reports make it very easy to identify & Reward top referrers.
  - Referrals are modeled on the Lead Object & all salesforce Lead Scoring are also present for referrals. (e.g. Einstein Lead Scoring, Lead Routing, Lead Assignemnt)
  - Referrals can be as easy as if someone is shifting to another location so he need some money for home but we don't give mortages so I will send referral to my friend & give her information so he will provide mortage to that person.
  - e.g.  
![referrals](referrals%20eg.png)

- We have to Modify some setting of our Org to use Referrals.
- It is best practice to assign referrals to a queue, so managers can assign each referral to the employee with the most time and experience. You can create an approval process to automate the way referrals are approved.
- After receiving the referrals the recipient can:  
    ● Accepts the referral and gets to work  
    ● Rejects the referral and sends it back to the referrer for more information  
- When there are many referrals in queue than they can use Salesforce Inbuilt Priority Tools.
- Referral Converstion : When a referral is qualified, it’s ready to be converted & Common qualifying steps include confirming client interest, completing all client information, or pre-qualifying the client for a mortgage.
- Your top referrers are a key part of your business. A referrer is someone who creates a referral, and your top referrers are the ones who create the referrals that are most likely to convert.
- The Referrals view shows details about the number of referrals, the number converted and rejected, the conversion rate, and a list of people referred. To keep your top referrers happy, they consider creating an incentive program to reward them with financial or other incentives.
- We can also track Track Referral Activity & Customize Referral Path.
- Actionable Relationship Center (ARC) : The ARC interface shows account, contacts, and related records in one view, letting users navigate among related records. ARC supports B2C and B2B relationships.
  - using ARC we can:
    - Get a 360 degree view of the client
    - Explore complex business hierarchical relationships
    - See related information such as Financial Accounts, Opportunities, Cases, etc.
    - Create new visualization UIs and associated experiences
    - Discover new areas of opportunity and influence
    - Take quick action on records (Add / Edit / Remove )
    - Expand hierarchy and Account relationships to allow group to group relationships
    - Model those Account to Account relationships
 
- The Actionable Relationship Center (ARC) interface lets you create, edit, and remove account-account and account-contact relationships. The ARC shows both Financial Services Cloud relationships and related list relationships in one view and it lets users navigate among related records.
- The Actionable Relationship Center (ARC) interface lets you create, edit, and remove account-account and account-contact relationships.
- To view an ARC graph, users must have read access to the fields on the Contact, Account, Reciprocal Role, Account-Contact Relationship, Account-Account Relationship, and Contact-Contact Relationship objects.
- ARC is supported for orgs that have implemented the Person account Model. ARC isn’t supported for orgs that use the Individual Model. ARC requires that the Association Type field of the Account-Account Relationship object has active picklist values with the following API Names: Group, Member, and Peer.
- ARC isn't available in the Salesforce mobile app.
- Record cards only show the first two fields that appear on the associated record’s compact layout.
- Custom actions can be added to record previews, but not to record cards.
-  Compliant Data Sharing : Compliant Data Sharing features make it easier to control who can access specific records. CDS works seamlessly with existing Salesforce data sharing features. It provides extra access rules, but otherwise does not override sharing behavior from existing features.