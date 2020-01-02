# What are the impacts of Test-Driven Development on code quality, code maintainability and test cover

## march 2017

## Authors

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Alexandre Cazala &lt;alexandre.cazala@gmail.com&gt;
* Nicolas Lecourtois &lt;lecourtoisn@gmail.com&gt;
* Lisa Joanno &lt;lisa.joanno@gmail.com&gt;
* Pierre Massanès &lt;pierre.massanes@gmail.com&gt;

## Introduction

This document presents the results of our researches on the Test-Driven Development method. In order to concretly present them, we present in a first section the context of our research. In the second section, we go deeper into the description of our study and on which project it is based.

### Research context

The **Test-Driven Development** \(TDD\) is a method of software development relying on **writing tests before the tested code** even exists and more importantly relying on refactoring code. More precisely, there are five different steps. First writing the unit test, then run the test to watch it fail. If the test succeeds, there is a problem since the tested code is not yet written. When the test is written and fails, the next step is to write just enough code to see the test succeed. Then, when the new test succeeds, the fourth step is to check that all the tests still pass. If there are some failures, it is necessary to fix the issues to have all the tests passing. Then the final step is to **refactor** the code in order to make it better. The Figure below illustrates the development process using the TDD method.![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/TDD_Global_Lifecycle.png/1024px-TDD_Global_Lifecycle.png)Figure 1 : TDD method process

The purpose of this method is to write the specifications first in the form of **unit tests** so the written code answers exactly to the wanted functionalities. More than that, it wants to assert that the code is always valid and more stable. It should also help the developer to avoid regression when refactoring the code.  
In this sub-chapter, we present our study of the impacts of the TDD method on code quality, code maintainability and test coverage. The study is splitted in three sub-questions:

1. Does TDD reduce the number of issues to fix during the development ?
2. Does TDD reduce the overall complexity of a project compared to the common Test Last \(TL\) method ?
3. Does TDD projects always have a high test coverage ?

Test-Driven Development promotes the fact of producing a code of better quality and always valid. This study aims to verify if this assertion is real or not and in which way. There is not yet an answer to this question which divides the developer community. We think that it could be interesting to compare this method to a more common way of developing, which is to develop functionalities first, then write the tests, to bring an answer to this question with concrete arguments. This common method is also known as the Test-Last \(TL\) method. Also, many companies do not approve this method thinking that testing first cost more than having something which works first. It would be interesting to see if it is true or not by comparing maintenance cost \(i.e, number of issues, fixes ...\) and productivity \(i.e, number of lines added and deleted\) in test-first method \(TDD\) with test-last method.

### Scope

In this study, we compare TDD projects and TL projects only. Any other development method is not part of this study, we made this choice in order to narrow our field of research. As our main question is about code quality, we don’t study the impacts of TDD on development time compared to TL. We have restricted our analysis to some metrics defining the code quality of a project and described in the part [Project Evaluation](what-are-the-impacts-of-test-driven-development-on-code-quality-code-maintainability-and-test-covera.md#project-evaluation). So external factors as team size, team experience or language used are not part of our scope. Even if these factors can have impacts on code quality and code coverage, we lacked time and resources to take them into account, they are part of the limits of this study.

Other concepts linked to the TDD method, like emerging software architecture, are not studied either, still in order to focus on answering our questions.

The projects studied here are either TDD or TL but how can we now the development method used in a project ? To choose the projects, we do not have an automated tool capable of detecting the development method used. We rely on the project team and the developer community. Developing following the test-driven method is not a common choice, so usually the project team clearly state that the project is test-driven. In addition to this, we also checked manually if the commits of the TDD projects seems to follow the pattern test first, then code. But this is just a partial verification, the best would be a tool analysing the commits and finding this pattern.

### Projects Studied

In order to find a concrete answer, we had to find many projects built using a TDD approach and of at least thousands of commits:

* [FitNesse](https://github.com/unclebob/fitnesse) is a project from Robert C. Martin, also known as Uncle Bob, who wrote many famous books about agile principles, code quality and best practices. It has more than 5000 commits.
* [JUnit](https://github.com/junit-team/junit4) was written by Kent Beck and Erich Gamma using TDD throughout. We study JUnit4 \(around 2,000 commits\).
* [JFreeChart](http://www.jfree.org/jfreechart/) is a 2D chart library for Java applications \(around 3,000 commits\).
* [OpenCover](https://github.com/OpenCover/opencover) is a code coverage tool for .NET \(around 1,200 commits\).

There are other projects using TDD but those are the most interesting for our works. We are limited in time by our studies.

We found the following TL projects which are approximately of the same size as the previous TDD projects.

* [Google Gson](https://github.com/google/gson) is a Java serialization/deserialization library that can convert Java Objects into JSON and back \(around 1,300 commits\)
* [JaCoCo](http://www.eclemma.org/jacoco/) is a Java code coverage library \(around 1,400 commits\)
* [Spoon](https://github.com/INRIA/spoon) is an open-source library to analyze, rewrite, transform, transpile Java source code \(around 1,800 commits\)

We compared TDD method projects with TL method projects based on:

* Cyclomatic complexity
* Many common code smells 
* Code coverage
* Issues
* Number of lines added per commit or per week
* Number of commits per week

### Expectations

As Test-Driven Development is really driven by the tests, we expect TDD projects to have a high code coverage, of at least 80% and higher than TL projects. This method involves an important refactoring phase, so cleaning the code is an important part of it. Because of this, we expect a better code quality but also more commits about refactoring and less about fixing or patching bugs.

## Project Evaluation

The previous section described our research context, our goal and the red string for our project. In this section we present how to evaluate those project samples \(i.e, in what and how we are evaluating those projects\).

### Metrics, a way to evaluate projects

In order to answer our three sub-questions we used many metrics. All of our TDD and Test-Last projects are evaluated using the same metrics to compare them.

#### Code age

The code age measures in month the last time a file has been modified. Because some of the projects are older than others, the real metric measured here is the average code age of each file relative to the project age. For example, if a file hasn’t been modified for 10 months and the project is 12 month old, we can say it’s a pretty much stable file. If most of the project contains files that are currently being modified it means that the developers have a lot of file to maintain. In the opposite case, it means that the developers just have to focus on a few files which is a sign of good maintainability.

#### Code coverage

The code coverage measures how much of a project has been tested.

#### Cyclomatic complexity

The cyclomatic complexity measures the number of paths through a function. Ideally there should be at least as much unit test as the cyclomatic complexity which should be as little as possible.

#### Code smells

Code smells are issues detected in the source code that can lead to a deeper problem. For example, Duplicated code, long methods and large class are code smells. The more code smells spotted in the source code, the more likely the project to be difficult to maintain.

### Evaluation process

During our projects, we applied the same process on seven different projects. We used four test-driven development projects and three test-last projects, as described before. We chose those seven projects of their likeliness in terms of commits and lines of code.

TDD projects :

* FitNess : 5 401 commits, 21 236 lines of code \(LOC\)
* JUnit4 : 2 160 commits, 4 682 LOC
* jFreeChart : 3 405 commits, 54 713 LOC
* Open Cover : 1 196 commits, 11 518 LOC

Test-last projects :

* spoon : 1 709 commits, 18 305 LOC
* GSON : 1 320 commits, 4 318 LOC
* JaCoCo : 1 365 commits, 11 740 LOC

For each project, we tried to obtain the data about the metrics we defined in the first part of the report.

#### JaCoCo

The first thing we did was configuring the JaCoCo \(Java code coverage\) for each project.

JaCoCo is a free code coverage library for Java. The advantage of using JaCoCo was the uniformity of generated reports, and its compatibility with Maven and Gradle. All of our studied projects use either Maven or Gradle. We configured the plugin for each project.

```markup
<plugin>
   <groupId>org.jacoco</groupId>
   <artifactId>jacoco-maven-plugin</artifactId>
   <version>0.7.5.201505241946</version>
   <executions>
       <execution>
           <goals>
               <goal>prepare-agent</goal>
           </goals>
       </execution>
       <execution>
           <id>report</id>
           <phase>prepare-package</phase>
           <goals>
               <goal>report</goal>
           </goals>
           </execution>
   </executions>
</plugin>
```

The Gradle configuration is similar. When the plugin is configured, you can generate reports with :

```text
$ mvn jacoco:report
```

For each project, we had a report like the following :  
![JUnit4 JaCoCo report](../.gitbook/assets/junit.png)  
Figure 2 : JUnit4 JaCoCo report

This report allows us to know the code coverage of each project, a metric we need to compare TDD and TL methods.

#### Sonar

We scanned each project with SonarQube. Sonar allowed us to get the general quality of the code. Like JaCoCo, the reports generated are the same for each projects, which allows us to compare the projects easily. To be able to scan with SonarQube a project, one needs to add a file called _sonar-project.properties_ to a project. The file we used for all our project is the following :

```text
sonar.java.source=1.8
sonar.sources=src/main
sonar.tests=src/test
sonar.junit.reportsPath=target/surefire-reports
sonar.jacoco.reportPaths=target/jacoco.exec
sonar.java.binaries=target/classes

#local props
sonar.login=admin
sonar.password=admin
sonar.host.url=http://localhost:9000

sonar.projectKey=JUNIT
sonar.projectName=junit
sonar.projectVersion=1.0
```

You need to have JaCoCo configured for your project \(previously described\) and your project built. After you launched the SonarQube server, you can scan your project with :

```text
$ sonar-scanner
```

The reports can be found onlocalhost:9000, where the list of all your projects will be displayed.

Another advantage of using Sonar is the uniformity of the generated reports. For example :![](https://github.com/RIMEL-UCA/Book/tree/bb02ad7c257a12ef511e6a2a2ce96a95f67d1db0/2017/assets/SonarQube%20junit.png)Figure 3 : Sonar reports for JUnit4

Using SonarQube was a way during our study to get the cyclomatic complexity of the projects, along with the sonar issues. Both are a metric we need to compare TDD and TL methods. Sonar defines a number of issues during a scan, for example bugs detected, vulnerabilities and code smells.

#### SoftVis3D

SoftVis3D is a framework to vizualize a project, litterally. It is available on the SonarQube update center. You need to install the plugin on Sonar and it is automatically available when you scan a code.

The goal is to provide a visualization for the hierarchical structure of a project. Folders or packages are shown as districts, files as buildings. The building footprint, height and color are dependent on two arbitrary sonar metrics : you can tell SoftVis3D wich value you want to use, and you can use any metrics. This tool is useful in order to have a complete view of a project, and to see if a god-class is present. For our project, we use it to judge the global cleanliness of a project. We used the complexity as footprint , the number of duplicated lines as height, and number of Sonar issues as the color.

![](../.gitbook/assets/junit-codecity.png)

Figure 4 : SoftVis3D results for JUnit4

#### Code maat

We used Code Maat as a tool to study GitHub repositories. Code Maat is a command line tool used to mine and analyze data from version-control systems. It allows us to perform many kind of analyses. Those in which we were interested are “age” \(the code age\) and “revisions” \(how many times a file has been modified\).

Running code-maat directly is not the most convenient way, so we made a simple cli in python with two commands : retrieve &lt;git\_url&gt; and analyse &lt;projects\_names&gt;. The first one clone the repository and run code-maat analysis on it, which gives as an output the raw data relative to code-age and revisions \(among other unexploited in our project\). The second one aggregates these raw data into readable statistics which we used to make our own analysis. It also counts the number of commits containing our predefined keywords \(fix, test and refactor\).

This script is in python and is available here:

[https://github.com/lecourtoisn/code-maat-cli](https://github.com/lecourtoisn/code-maat-cli)

Here is an example of output we used :

```text
fitnesse       fix 6%(365)   , refactor 1%(87)    , add 4%(234)   , test 14%(809)
origin         fix 9%(1684)  , refactor 1%(194)   , add 6%(1111)  , test 9%(1604)
spoon          fix 30%(535)  , refactor 7%(126)   , add 13%(243)  , test 12%(223)
jacoco         fix 2%(35)    , refactor 0%(2)     , add 2%(39)    , test 11%(151)
junit4         fix 7%(156)   , refactor 1%(24)    , add 5%(115)   , test 14%(303)
node           fix 17%(2850) , refactor 2%(383)   , add 17%(2883) , test 18%(3084)
gson           fix 6%(92)    , refactor 0%(8)     , add 6%(83)    , test 13%(183)
jfreechart     fix 1%(49)    , refactor 0%(1)     , add 3%(104)   , test 7%(266)
```

## Study

For each project, we applied the previously described evaluation process, except for the SonarQube analysis of OpenCover which can be found directly online. In the next two parts we present and analyse these results to answer our questions.

### Raw results

Here are the raw results we obtained after analysing the projects.

#### SonarQube and CodeMaat results

| Metrics | Test-Driven Development |  |  |  | Test-Last Development |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | Fitnesse | JUnit4 | JFreeChart | OpenCover | Spoon | GSON | JaCoCo |
| Code Coverage | 48% | 85% | 45% | 93.9% | 90.7% | 83% | 80% |
| Sonar issues | 1927 | 833 | 5039 | 286 | 2341 | 592 | 200 |
| Complexity | 8612 | 2061 | 19323 | 1568 | 7635 | 1945 | 1962 |
| Code Age | 48.3% | 21.5% | 18.1% | 83.6% | 9% | 50% | 35% |
| Average number of reviews/files | 4.46 | 7.72 | 4.97 | 3.15 | 6.35 | 17.26 | 9.12 |
| % "Fix" Commit | 6% | 7% | 1% | 7% | 30% | 6% | 2% |
| % "Refactor" Commit | 1% | 1% | 0% | 2% | 7% | 0% | 0% |
| % "Test" commit | 14% | 14% | 7% | 9% | 12% | 13% | 11% |

#### SoftVis3D results

#### TDD projects

![](../.gitbook/assets/fitnessebobcodequality.png)Figure 5 : Fitnesse

![](../.gitbook/assets/jfreechart-codecity.png)

Figure 6 : JFreeChart

![](../.gitbook/assets/junit-codecity%20%281%29.png)

Figure 7 : JUnit4

#### Test-Last projects

![](../.gitbook/assets/gsoncitycodequality.png)

Figure 8 : Google GSON

![](../.gitbook/assets/jacococodequality.png)Figure 9 : JaCoCo

![](../.gitbook/assets/spoon-codecity.png)

Figure 10 : Spoon

### Critical Analysis

From the raw data we had collected and for each of our metrics, we made charts to obtain a better visualization and make comparisons easier. The four first projects in the charts are TDD and the three last TL.

![](../.gitbook/assets/test_coverage.png)

Figure 11 : Test Coverage

This chart represents the percentage of code coverage for each project. We can see that TL projects have a code coverage higher than 80%, so they are mainly well covered by tests. For the TDD projects, there is some disparity. Two projects have a code coverage higher than 80% but the two others have a coverage of 48% and 52%, which is really low. We expected TDD projects to have a high code coverage, but our study shows the opposite. TL projects seems to have a better code coverage than TDD. But as we studied just a few projects, these results can be just exceptions or the consequences of external factors.

![](../.gitbook/assets/count_keywords.png)

Figure 12 : Proportions of commits

With this view, where the projects are kept separated, we can see that the results are more or less uniform. Except for Spoon, whose number of fix related commits is surprisingly high, about four times higher than the other.

![](../.gitbook/assets/compare_keywords.png)

Figure 13 : Proportions of commits. TDD compared to TL

As expected, using a Test-Last method implies to have more fixes \(and so, have more bugs\). However, Spoon has falsified our data, it proves that we need more data to have a concrete representation of the Test-Driven Method and the Test-Last method. Also, the first metric shows that in Test-Driven Development we got less tests than in Test-Last. This metric is only a study of their commits and not representing the test coverage. Yet, due to the way TDD works, with phases of test and phases of refactoring, we expected TDD projects to have a higher percentage of refactor commits.

![](../.gitbook/assets/mean_revisions.png)

Figure 14 : Stability of files

The first four are TDD projects and others are TL projects. GSON is a TL project made by Google, this statistic shows that each file has been edited approximately 17 time \(in average\). We know there is a correlation between the number of revisions and the number of bugs proven by many researchers \(Thomas Zimmermann studied it in its researches : [https://goo.gl/eNVqAK](https://goo.gl/eNVqAK)\). So it seems that GSON, based on this metric, isn’t a clean project.

Globally, TL projects are less stable \(more often edited\) than TDD projects and consequently more subjects to bugs.

![](../.gitbook/assets/compare_metrics.png)

Figure 15 : Stability and Test Coverage \(TDD compared to TL\)

This graphic shows three important metrics. Globally, we can see that files in TDD projects are more stable and, consequently, less exposed to bugs. However it is interesting to see that the mean of TDD test coverage is under the mean of TL test coverage projects. Using the TDD method doesn’t mean to have more test, it is just a method where you have to write tests before the code.

Visualizing the SoftVis3D representations, where the base is the cyclomatic complexity, the color is the number of sonar issues and the height the number of code duplication, it appears that projects using TDD have a lower cyclomatic complexity than TL projects, which is what we expected. This results may be a consequence of the process explained by Kent Beck during a TDD project :

> The two rules imply an order to the tasks of programming:
>
> * Red—write a little test that doesn’t work, perhaps doesn’t even compile at first
> * Green—make the test work quickly, committing whatever sins necessary in the process
> * Refactor—eliminate all the duplication created in just getting the test to work
>
> Red/green/refactor. The TDDs mantra.

The refactor phase happens after every task during TDD, so developers are probably more used to refactoring, and the quality of the said refactor may be higher. A higher refactor quality may explain the lower cyclomatic complexity in TDD method, because developers devote more of their time to refactoring.

## Conclusion

In our study, we investigated the impacts of Test-Driven Development on code quality and code coverage compared to projects following the Test-Last method. We analysed seven projects of similar size to evaluate these impacts. The results that we analysed shows that the TDD projects have an overall complexity and code quality better than TL projects, according to our expectations. However, the results gathered about code coverage are not matching our expectations. Half of the TDD projects have a code coverage lower than 60% and all the TL projects have a coverage higher than 80%.

As we analysed just a few projects without a precisely defined context, we cannot generalize the results obtained beyond our scope. Still we can see that our results about code quality are similar to the ones of the study made by Bhat and Nagappan who also studied the impacts of TDD on development time \(which is not part of our scope\). So we hope that this study will contribute to the research in this field, especially about the confidence on the impacts on code quality while using the Test-Driven Development method.

## References

### Projects

* Uncle Bob website: [www.cleancoder.com/](http://www.cleancoder.com/)

TDD projects:

* Fitness: [https://github.com/unclebob/fitnesse](https://github.com/unclebob/fitnesse)
* JUnit4: [https://github.com/junit-team/junit4](https://github.com/junit-team/junit4)
* JFreeChart: [https://github.com/jfree/jfreechart](https://github.com/jfree/jfreechart)
* Open Cover: [https://sonarqube.com/dashboard/index?id=opencover](https://sonarqube.com/dashboard/index?id=opencover)

TL projects:

* Spoon: [https://github.com/INRIA/spoon](https://github.com/INRIA/spoon)
* Google Gson: [https://github.com/google/gson](https://github.com/google/gson)
* JaCoCo: [https://github.com/jacoco/jacoco](https://github.com/jacoco/jacoco)

### Articles

Beck, K. \(2003\).Test-driven development: by example. Addison-Wesley Professional.

Dave Astels. \(2003\).Test Driven Development: A Practical Guide. Prentice Hall Professional Technical Reference.

M. Pancur, M. Ciglaric. \(2011\).Impact of test-driven development on productivity, code and tests: A controlled experiment. In Information and Software Technology 53 \(pp. 557–573\)

Bhat, T., & Nagappan, N. \(2006, September\). Evaluating the efficacy of test-driven development: industrial case studies. InProceedings of the 2006 ACM/IEEE international symposium on Empirical software engineering\(pp. 356-363\). ACM.

Martin, R. C. \(2008\).Clean Code: A Handbook of Agile Software Craftsmanship. Pearson Education.

Kaufmann R. & Janzen D. \(2003, October\). Implications of test-driven development: a pilot study. In Companion of the 18th annual ACM SIGPLAN conference on Object-oriented programming, systems, languages, and applications \(pp. 298-299\). ACM.

Bhat, T., & Nagappan, N. \(2006, September\). Evaluating the efficacy of test-driven development: industrial case studies. InProceedings of the 2006 ACM/IEEE international symposium on Empirical software engineering\(pp. 356-363\). ACM.

Thomas Zimmermann, Nachiappan Nagappan, and Andreas Zeller, Predicting Bugs from History. [https://goo.gl/eNVqAK](https://goo.gl/eNVqAK)

