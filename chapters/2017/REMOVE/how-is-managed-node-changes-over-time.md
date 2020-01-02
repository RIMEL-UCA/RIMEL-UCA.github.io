# **How is managed node changes over time?**

## **Case Study**

In the world of software management, dependency management is a hell. The system evolves more it integrates components, most system management is becoming increasingly complex.

Nodejs experienced strong growth in recent years. This includes the number of modules published the number of contributors. Node is a platform with many dependencies, publish a new version of a component can be a nightmare very quickly.

![](/assets/Capture.PNG)



In this chapter, we will focus on the development of nodejs philosophy

## **Methodology**

We will start by watching commits over a given period, identify which modules evolve together over time and different contributors to these modules.

Then we will analyze:

* the dependency rules between these correlated modules

* * are they strict / loose ?
* the version of a module changes

* The contributions from developers on a module

* * Who is working on such a module

  * Which developer commit the most lines

  * What are contributors to module X Version X.Y.Z

## **References**

[Foundation node](https://nodejs.org/en/foundation/)

[Registry npm](https://www.npmjs.com/package/npm-registry)

[Semantic versioning](http://semver.org/lang/fr/)

[A Unified Framework for the Comprehension of Software's time Dimension](https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/11998/Benomar_Omar_2015_these.pdf?sequence=2&isAllowed=y)

## **Tools**

[CodeCity](https://wettel.github.io/codecity.html)

[Github ap](https://developer.github.com/v3/)

[Gitinspector](https://github.com/ejwa/gitinspector)

## D**istribution of work in the team**

* **Balde Thierno**

  * Commits analysis over a given period

  * Identification modules and correlated these contributors

* **Diallo Mahmoud**

  * Versions Analysis of Module

  * Contributions Analys is different developers on a given module

**  
**

