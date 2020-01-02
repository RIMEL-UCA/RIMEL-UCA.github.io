# What about **Architecture & Design ?**

The goal is to use static analysis of code to find interesting results about each feature, be it caching or dialects, and, if possible, to generalize our methods to be used in other scenarios

## **Tools used :**

* Codescene

* IntelliJ UML plugin: generating UML from code and exporting it as XML

## **Methodology :**

Through this question we’ll try to take a closer look at the different caching strategies used in both frameworks, by having access to the source code, and more specifically the parts dealing with cache, we hope that, through code analysis, we will be able to find the differences \(or the similarities\) in this area.

We will also take a look at the classes responsible for generating the SQL for the different dialects

**References : **

* [Reverse Engineering Java Code to Class Diagram: An Experience Report](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.259.546&rep=rep1&type=pdf)

## **How **does **Entity Framework and Hibernate manage d**ialects**? **Are there any major differences?

## Dialects

let's start with hibernate:

Let’s start with dialect: we used simple search functions, such as “find” and “grep”, to search for key words, the idea being that, by searching for those words, they will leads us to java classes that implement our features, so the search for the word dialect lead us to a package containing different classes:

* ·A mother classes containing various SQL key words and functions

* ·A more interesting result : we found classes representing different database products : postgres, MySql …  
  this is very important, as it gives an insight into dialects supported by hibernate

![](/assets/uml_postges.png)

Figure : generated UML for Dialect package

As we can see, each new version extends the older one, to alter its behavior or to add something new.

what about Entity Framework ?

using our usual method \(key work search\), we didn't find any dialects dedicated to known databases such as postgres or MySQL,

SQL Server is the mantra, EF supports it and nothing else, there are some connectors to use postgresql with EF but nothing official

This is rather a big difference between the two frameworks

To get more insight into the sql generation, we explored a little bit the classes we found:

![](/assets/AST.png)

Figure : SQL generation logic

this is how it goes, we go through the DbContext \(which is the equivalent of session on hibernate side\),  calling a fetch method, the fetch method targets the cache, we have then a cache default, the id of the class of the entity and its id are delegated to the Linq which is an internal SQL DSL \(Domain Specific Language\), it builds the SQL query using the entity type and its id, the LinqAST parser parses the AST generating SQL Server compliant query and send it to the database.

## **How **does **Entity Framework and Hibernate manage caching? **Are there any major differences?

Moving on to caching, finding interesting information was a lot harder compared to dialects.

let's start with hibernate:

We used the same technique: searching by key word “cache”, we were able to find an interface defining various cache-related methods, but this leads us nowhere.

The next thing we tried is taking a look at the class that talks to the database:”SessionImpl”

We used our IDE, and generated an UML to see if there are any references to cache classes:

![](/assets/UML.png)Figure : generated UML for SessionImpl

As we can see, no references to cache

So the next thing we tried, is to examine the method that loads entities from the database, the reason being, that any cache related logic **has to be**, contained in this method. Let’s call it “the load method”

And we were not disappointed,

We found out, that en event approach was used, when the load method was called, a number of listeners are triggered, this explains why UML didn’t give us useful information, because references to those listeners are contained within methods, as opposed to a more classical OOP mechanism, such as inheritance and composition, which are more visible in UML

By examining the event listeners we found that, they also have a load method in which we can see:

![](/assets/code.png)Figure : Calls to first & second level cache

This gives a very important insight, as we learned that hibernate has two levels of cache

Granted, the analysis of caching is more manual that it is static, but we learned that the hard way, it is very difficult to find information about a complex feature without resorting to manual work and reading the code, for various reasons, even the programming approach can affect greatly the efficiency of static analysis, as we mentioned before, the fact that an event driven approach is used, made it impossible to see references to cache classes in the generated UML

What about Entity Framework ?

Entity Framework provides caching but only first level one, since the stateManager is specific to DbContext \(but we don't have specific one on the DbContextFactory level\), so a DbContext caches only entities that it fetched so it cannot see entities cached by another instance of DbContext, let's see how it works :

![](/assets/Caching.png)Figure : Calls to first level cache

when we try to fetch entity by identity key from the DbContext, the stateManager has an instance of IdentityMap, this class has basically a property which is a Dictionary that maps keys to entities, the entity is being fetched from there if already cached

## Conclusion

Our goal was to see to what extent can we  use static analysis to compare hibernate and Entity Framework, this was relatively easy or hard depending on various factors: the feature in question \(dialects vs caching\), the product \(entity framework vs hibernate\) ...

In general it is extremely hard to extract feature related information from the code, a lot of the work is rather manual, but despite the difficulty we managed to find some fundamental differences between the framework

Static analysis can be useful as an entry point to do analysis, it gives a broad idea about the feature in question, but to go in depth, one has no choice but to resort to reading the code

