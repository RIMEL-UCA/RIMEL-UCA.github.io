# What Conway's law is talking about ?

Conway’s law it's a sociological based predicate, deduced by observations but that seems to be true in most of cases. He said for **example **:

> _If you have four groups working on a compiler, you'll get a 4-pass compiler_

The bigger an organization is, more social boundaries are produced: majors are caused by communication problem.

Developers are not robots and they can’t work together perfectly because humans can’t share their mind: that’s why there is a conception phase before developing something. Working on the same code part means that you spend time on integrations and ensuring that the integration will be good enough, developers need to speak together. Sometimes the project are so big that we need a **big team**, dispatched all around the world and communication can be hard.

Figure 1![](/assets/ConwaysLaw.png)

Generally, to avoid this lack of communication, organizations which design a system are constrained to produce designs which are copies of the communication structures of these organizations. The** Figure 1 **illustrates this problem : the entire system is based on a design, and the developers team have to reflect the design structure.

### _**The Goal ?**_

If companies try to split the work between developers by adding some rules based on that Conway’s , we can imagine the job can be done **better and faster**.

When you ask to a team of three developers to solve an easy problem such as creating a simple website: The first says “I know Node.js, you can simply add a server.js file to route all requests”, while the second will say “Java allows you to create simple http-servlets to create a server” and the third will say “I don't know backend programming, but there are a lot of online hoster to create a simple http server”. Each specialist finds the way to answer to the question as it own, but with several ways.

To avoid comprehension problems between team members, software architects are setted up to clarify which way to take in development. If the architect is far from the his team \(eg. 2-3 hours jet lag\) that can slow up the subject comprehension. Generally, the developers who share the same module editing, must talk each other to clarify in which way progress: as you can see, proximity is the key and call reachability is essential.

Spring’s team is well dispatched around the world, we want to determine if we can use the Conway’s law to deduce the ideal team and compare it to get a better coding performance.

### _**How to ?**_

According to the Conway's law, the ideal team is a team where all the members whose develop part of a project are gathered in the same place. And if several geographical groups are formed, then the best thing is for each geographic group, to work on a small part of the project in order to limit problems due to remote communication.

We will take two approaches :

* The first one try to be more analytic on the actual Spring's team. To be more specific, on a module of Spring project \(we will talk about modules and submodules in the next part\).
* The second one is a bottom to top approach : we start from code analysis and we end by creating an ideal team based on some metrics in accord to Conway's law.
* The final part will consist to gather all that results and trying to prove the suitability of Conway's law.

### _**References :**_

* Alan D. MacCormack, John Rusnak & Carliss Y. Baldwin \(2014\) , Exploring the Duality between Product and Organizational Architectures: A Test of the Mirroring Hypothesis  
  , \[[http://hbswk.hbs.edu/item/exploring-the-duality-between-product-and-organizational-architectures-a-test-of-the-mirroring-hypothesis](http://hbswk.hbs.edu/item/exploring-the-duality-between-product-and-organizational-architectures-a-test-of-the-mirroring-hypothesis)\]

* Sam Newman \(2014\), Demystifying Conway's Law \[[https://www.thoughtworks.com/profiles/sam-newman](https://www.thoughtworks.com/profiles/sam-newman)\]

* Allan Kelly \(2013\) Conway's law v software architecture Published at DZone with permission of Allan Kelly, DZone MVB.

* N. Nagappan, B. Murphy, and V. Basili. International Conference on Software Engineering, Proceedings \[online\] \(visited on 18/12/2016\). \[The Influence of Organizational Structure on Software Quality\]

* Frank Philip Seth. The Influence of Organizational Structure On Software Quality: An Empirical Case Study \[online\]. \(visited on 18/12/2016\). \[[https://www.microsoft.com/en-us/research/publication/the-influence-of-organizational-structure-on-software-quality-an-empirical-case-study](https://www.microsoft.com/en-us/research/publication/the-influence-of-organizational-structure-on-software-quality-an-empirical-case-study)\]  
  \([https://www.microsoft.com/en-us/research/publication/the-influence-of-organizational-structure-on-software-quality-an-empirical-case-study/](https://www.microsoft.com/en-us/research/publication/the-influence-of-organizational-structure-on-software-quality-an-empirical-case-study/) "[https://www.microsoft.com/en-us/research/publication/the-influence-of-organizational-structure-on-software-quality-an-empirical-case-study/](https://www.microsoft.com/en-us/research/publication/the-influence-of-organizational-structure-on-software-quality-an-empirical-case-study/)"\)

* Lappeenranta University of Technology, LUT. "Human and organizational factors influence software quality." ScienceDaily. ScienceDaily, 11 August 2015. \[online\]. \(visited on 18/12/2016\). [www.sciencedaily.com/releases/2015/08/150811091913.htm](http://www.sciencedaily.com/releases/2015/08/150811091913.htm)

### Authors _**:**_

* Manuel PAVONE, SI5 \(5th year in Computer Science\) at the engineering school Polytech Nice-Sophia.

* Dorian BLANC, SI5 \(5th year in Computer Science\) at the engineering school Polytech Nice-Sophia.



