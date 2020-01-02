# How to do feature-driven comparisons ? application to ORMs.

Object-relational mapping \(ORM\) is a mechanism that makes it possible to address, access and manipulate objects without having to consider how those objects relate to their data sources. It abstracts away the actual details, ORM lets programmers maintain a consistent view of objects over time, even as the sources that deliver them, the sinks that receive them and the applications that access them change.

ORMs are very widely used in large scale enterprise applications, they take care of important aspects of the database access such as caching strategies and connection management \(pooling …\), which can have quite an effect on the overall performance of the application

Through this question we want to take a closer look at ORMs, Hibernate and Entity Framework, the purpose of this study is not to find out which ORM is better, but rather to use reverse engineering in order to extract interesting information about the two frameworks, in terms of \(a\) organization of their respective teams, and \(b\) two features of our choosing : Dialects and Caching



We find This comparison interesting because we are looking at two products that do the same thing but come from different mentalities, entity framework is a Microsoft product and only came to the open source world recently \(2013\), so it is interesting to compare the two products.

**Team :**

* Buisson Kevin

* Dahmoul Salah

* El Amrani Achraf

* Tijani Yassine



