# What about Organization ?

In this chapter we’ll be interested in the organizational differences between the two ORMs, to do so, we’ll examine the implementation of two features of our choosing: Caching, and dialects, the reason behind this choice is the fact the we find them interesting and also to make this study a bit more specific, given that ORMs cover a big variety of functionalities,

To do so, We will try to answer the following questions :

## General organization of hibernate and Entity Framework core

First of all we'll present the two communities working on the projects, Hibernate is an open source framework led by a core team that works on Red hat, the following picture shows the interactions between developpers on the hibernate project :

![](/assets/lead.png)Figure : Communication between hibernate team members 

As we can see from the highlighted paths, Steve Ebersole talks pretty much with the entire team, so there is a good chance he is the project's lead\(we confirmed this by going to his stackoverflow account\), in addition there are two big contributors, which gives to the project a clear direction, as the vision is not shared across several developers, the flip side being that if the lead \(i.e Steve Ebersole in our case.\) leaves the project, there will be huge knowledge loss \(unless there's a transition period\).

on the Entity Framework side, things are a bit different, the framework is a part of .NET core stack \(newly open sourced\) which is maintained by Microsoft, the following picture shows the interactions between developpers on the project :

![](/assets/leadEntity.png)Figure : Communication between Entity team members

Here we can see that there's no established lead on entity framework as the half on the right is made of leads, in this case we have a knowledge sharing, but different visions as each one of them may have a different point of vue about where the product is heading.

So we can already see a divergence between the two frameworks with respect to organization, this result is general, so let's see what happens when do a more precise comparaison, this time well lock at the features :

## Organization around dialects and caching on hibernate and Entity Framework core

when doing feature driven comparison, it might be interesting to look closely at the team organization around the targeted features, as sometimes it may give us insights about feature importance to the team, stability and maintenance.

### Dialect organization on Hibernate

Using codescene, and based on number of commits, we can see the most frequent contributors in a given class, each class is represented by a circle, the size of a circle represents the class's size \(in terms of lines of Code\) :

![](/assets/ownership.png) Figure : Ownership of Dialect Classes in hibernate

we can see that Hibernate supports a variety of SQL dialects, we can also see that Steve Ebersole is the dominant contributos \(most circles are red\), which we makes since, since dialects are a core feature that should be present since the early days of the framework, so it is not surprising that the lead developer has the ownership.

### Dialect organization on Entity Framework

![](/assets/dialect_entityFramework.png)Figure : Ownership of Dialect Classes in Entity

in contrast to hibernate, we can see that ownership of the dialect feature is dispatched through the team, which may slow the

development since it will increase interactions and coupling between developpements.

We can also see that the general organisation is maintained when we specialse our stugy to specific features

### Caching organization on Hibernate

The same principale applied to caching classes :

![](/assets/cache_feature_h.png)Figure : Ownership of caching Classes in hibernate

here we have the general cache structure package, which owned again by hibernate's lead, we can assume that this behaviour may be a pattern that is reproduced on all features.

### Caching organization on Entity Framework

![](/assets/cache_feature_en.png) Figure : Ownership of caching Classes in hibernate

On Entity side, we can see also a fair repartition between the leads, once again, the same pattern is kept .

## 



