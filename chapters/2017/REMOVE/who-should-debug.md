# I. Who should debug ?

## Problematic

![](/assets/RIMEL_IntroBis.jpg)

During the maintenance of a project, we can issue several concerns. One of them is debugging features, which can take some time and be done more or less well

But how to know who we should attribute this task to?

That's the question we'll try to answer with this study.

We would like to obtain two results:

* Obtain the result for a specific team composition and on an individual point of view, meaning :

  * Should X debug the feature he developed?
    * Yes because he generally does it well and fast
    * No, and in this case, who should do this task?

* If this leads to a concluant result, we could extend our study to a general statement, meaning:

  * Should we attribute debugging tasks to newcomers / advanced developers other than the author or the author himself?

And also make more specific statements :

* Newcomers tend to \(or not\) generally debug slower than other developers but do that better \(in code quality way\).

## Definitions

Code Quality : General state of the code regarding the metrics linked to this notion at a moment.

## Process Description

![](/assets/RIMEL_ProcessHead.jpg)

We are basing our study on Open Source projects, and more specifically ElasticSearch as a base project, but by applying our methodology, you can obtain the results for your own team or project.

We chose this project because it provides all informations about the differents issues on this project \(meaning the issues with the label bug that were corrected, and all the commits linked to this issue\), and have a project that has been existing for a long time, with developer turnovers, and a large number of contributors.

![](/assets/RIMEL_ProcessDetail.jpg)

Our process is splited in 4 differents phases that will be detailed below, let's just explain the general purpose of each phase :

1. The Listing phase is here to identify a corrected bug and retrieve all the commits related to this issue through a manual parser, and save those informations
2. The second phase is here to identify the author / corrector situation meaning who introduced the bug and who corrected it? 
3. The third phase is here to compare the code before and after the code corrections by comparing different metrics that will be detailed below. The final goal is to generate a report detailing all the informations about this comparison
4. The forth phase is here to compute all the results provided by the 3 precedent phases and generate a final report that will display the final results of our study on this project.

## Tools and Methodology

### Phase 1: Identify the bug and the commits concerned

![](/assets/RIMEL_Phase1.jpg)

#### Tools used:

API Github

#### Detailed methodology:

We start by listing all the issues that were created on the repository. This can easily be done with the Github API  
[https://developer.github.com/v3/issues/\#list-issues-for-a-repository](https://developer.github.com/v3/issues/#list-issues-for-a-repository)  
We just need to add a filter that let us retrieve only the issues with the label “bug”.

Result: We now have the list of all the issues that concern bugs and bugs correction for this repository.

###### We now look for who worked on debugging this feature, and which commits of the git repository are concerned by this debug.

In order to do this, we once again use the Github API, which allows to retrieve all kind of events for a given issue.  
One of them is particularly interesting in our study: CommitCommentEvent  
They give us the information about all the commits that are linked to the debug of this feature, and informations about their authors.  
If we look at the first commit of this list, we can easily know the state of the Git when the debug was implemented \(we just need to go to the commit N-1\).

![](/assets/Github Issue.png)

#### Result:

We now have the commit we have to start our study from.  
And also the list of the modifications and their authors made in order to debug this feature.

### Phase 2: Identify the author / corrector situation

![](/assets/RIMEL_Phase2.jpg)

#### Tools used:

Git functionnalities and scripts

##### We are now looking to identify the current “author case” we are in, meaning:

* The author of the feature corrected his own bug_ \(@mbf : mais comment vous affectez un feature à un auteur? Quelle approximation?\)_
* An other developer corrected his feature

We can also extend the “other developer” by newcomer, by completing our methodology.  
This also leads to other questions like “From when can we consider that our developer is no longer a newcomer?”, that we won’t answer in this study.

In order to do this, we need to use a code comparison tool.  
This can be done using different methods:  
We retrieve the concerned files and use gumtree between those files.  
We use git blame that also provide this functionnality, but directly on the concerned git.

We here chose the second solution because it was faster to obtain decent results.

To do that we need to clone the repository in our local storage, and then use the git blame command, but within the commits concerned by our correction.  
This can be done with this command:

> git rev-parse ^\[FIRSTCOMMIT\] \[LASTCOMMIT\]  \| git blame -S /dev/stdin \[FILECONCERNED\]

\[FIRSTCOMMIT\] : Needs to be replaced by the ID of the first commit event that we identified before.  
\[LASTCOMMIT\] : Needs to be replaced by the ID of the last CommitEvent that we found  
\[FILECONCERNED\] : Needs to be replaced with the file concerned by our study \(all the files that were edited to correct this bug\).

We can also edit the output buffer by editing the /dev/stdin part.

With this we know who edited which line, if the line starts with ^, it’s because it wasn’t edited during our bug corrections, if it was, we have the commit that edited this line, and it’s author.

We need to compare between the author of the initial line and the final line

#### Result:

We now know the case we are in \(Self Correction / Someone Else correction\).

### Phase 3: Analysis of the code quality

![](/assets/RIMEL_Phase3.jpg)

#### Goals:

The goal of this phase is to generate a report that will be used to compare the quality of the code before and after the debugging task.

We can easily retrieve the pre-debugging state of the code by pulling the commit before the first one in our list of commits for this issue. We can then the final state of the code by pulling the last one in this list.

We are trying to identify two general values during this phase :

* A general debugging score that is a combination of the different metrics.
* A detailed report that will be used during the final phase in order to compare the studied metrics for each issue.

#### Tools used:

Shell scripts  
Jacoco  
Code-maat

PMD / checkstyle

#### Compare Cyclomatic Complexity

After a lot of research we found out two tools that could possibly match our requirements:  
PMD that can be found here : [http://pmd.sourceforge.net/pmd-4.3.0/running.html](http://pmd.sourceforge.net/pmd-4.3.0/running.html)  
and checkstyle here: [http://checkstyle.sourceforge.net/cmdline.html](http://checkstyle.sourceforge.net/cmdline.html)

These are both command line tools that allows us to retrieve the cyclomatic complexity and other metrics.  
Those are tools that are mostly used to detect problems within a project, so they might not be the most  appropriate to just detect differences between two versions of a project \(but could definitely be a plus if we want to calculate more code quality criterias\).  
We could for example use them in order to know if we still match our base criterias \(Maximum complexity, maximum LoC, code coverage …\) after the debug phase.

_**For checkstyle:**_

In order to retrieve the cyclomatic complexity with this tool , we need to generate a configuration file, and then add the cyclomatic complexity metric.  
More informations about it can be found here:  
[http://checkstyle.sourceforge.net/config\_metrics.html](http://checkstyle.sourceforge.net/config_metrics.html)

_**For PMD:**_

In order to calculate the cyclomatic complexity, we can refer to this part of the documentation:  
[http://pmd.sourceforge.net/pmd-4.3.0/rules/codesize.html](http://pmd.sourceforge.net/pmd-4.3.0/rules/codesize.html)  
We just ned to apply those rules to our configuration file in order to retrieve the results expected.

#### Compare Number of Lines of Code

Under linux, we can easily retrieve the number of lines in a file with the command:

> wc -l \[FILENAME\]

We just need to do this for both files and store it.

#### Compare dependencies

Code-maat helps to calculate the logical coupling within the same application:  
[https://github.com/adamtornhill/code-maat](https://github.com/adamtornhill/code-maat)

> java -jar code-maat-0.9.0.jar -l logfile.log -c git -a coupling

We also need to compare coupling with external libraries.  
One way to actually do this is to checkout if one of the commit adds an import line in the top lines of the file or edits the packaging system in order to add new dependencies.

#### Code Coverage

The fact that we had a bug that was not detected means that we potentially didn’t test this feature enough.  
A good point to explore in addition to the previous one is the code coverage for this feature after the debugging happened.

Adding test after the end of the debugging helps us to improve the quality of the software on a known problem.

Those tests can be done with jacoco which can generate reports about the code coverage in both xml and csv format.

The tool can be found here: [http://www.eclemma.org/jacoco/](http://www.eclemma.org/jacoco/)

### Phase 4: Results computation

![](/assets/RIMEL_Phase4.jpg)

This final phase is ment to compute all the results given by the Phase 1 to 3.

For each issue, we will obtain a report that will include :

* Who introduced this bug
* Who corrected it
* What is the relation of the corrector \( Newcomer on the Project / Advanced developer / Author himself\)
* Metric score
* Link to a more detailed report \( Phase 3 datas\)

With this report, we'll be able to then generate a final report that will assemble all the datas and give a final answer to the question "Who should debug? " in the scope of our study.

_**Note :**_ We don't need to study if a correction introduced new bugs since the ownership of the code will change with this task, so our recursive study of the project will automatically detect this problem.

We here take the assumption that if someone corrects a bug on a portion of the code, then he totally takes the ownership of it, meaning that if he didn't corrected all bugs on this portion and leaves some after his actions, he'll then be responsible of them.

A first approach to the answer is to simply check and add the results given for each report.

We'll obtain a chain of scores like::

* Programmer 1 : Score 230 as Newcomer, 50 as Advanced Programmer, -50 as Owner
* Programmer 2 : Score 10 as Newcomer, 30 as Advanced Programmer, 10 as Owner

A positive score means that a programmer adds value to the code he touched regarding our generated metric.

A negative score means that the programmer affected the code in a bad way with his corrections.

We will have a score for each of the ownership / experience case in this project.

The newcomer score will generally reflect the quality of his work as a newcomer in the company, debugging other people bugs.

The Advanced Programmer score will reflect the quality of his work once he acquired some experience on the project.

The Owner score will reflect the quality of his own work, meaning introducing bugs and correcting them.

We can then produce a more detailed report, by combining the informations of the detailed report, meaning we can also know

* Programmer 1 : 
  * Owner:{ Code coverage test : 10, Code Complexity: -20 }
  * Newcomer { Code coverage test: 40, Code Complexity 20 }
  * Advanced { Code Coverage test: -20, Code Complexity: 30 }

This allows us to have more detailed informations about the code produced by this coder, meaning he generally add tests coverage while debugging his code but also adds complexity to his code, improves code complexity but don't add tests on a large number of case when debugging other people's code as an advanced programmer on a project ...

## Identified Problems:

We found several problems in our methodology that we need to fix:

We currently make the assumption that the only edits done to a file between the initial commit and the last debug commit are only bug correction, which is not the case.

In order to improve this element, we should develop a process that would need to check each commit, save the line edited, and then use one of the 2 solutions:

Start from the pre-bug version of the file and apply the corrections to it, and then calculate all the things we wanted \(with this solution, all other edits on the file would be rollbacked so we don’t consider them when analyzing our code\).  
Find a way to analyse only the part of the code we want to \(Calculate lines added/removed only in the bug correction sections / calculate code complexity only for the methods concerned …\)

1. We currently Consider that there is only one person correcting a bug, but how should we consider our study when multiple persons are working on a bug correction?

2. We have a large amount of interactions with external API’s and command line tools, so we need to synchronize everything through a script that also has to implement a large amount of parser in order to retrieve only the needed informations.  
   Moreover, we are here explaining the command chain for a single bug, but in order to obtain some results we need to apply this to all the bugs of the repository and then generate some statistics with our results.

### Extra Tools:

JArchitect is mentioned in many topics about tools that could help us on the last part, and looks like a great tool that is a fusion of all of those used in the third phase of our study.  
However this tool isn’t free to use, so you may want to use it if you already have a license or if you can afford it.

## Further Questions

### Code Ownership

How to improve code ownership algorithm to be more relevant ?

### Relevant Metric

How to compute metrics to have a relevant score?

### Bug Difficulty

Debugging easy bugs may be less rewarding but how to qualify the difficulty of a bug?

### Multiple Correctors

How to adapt algorithms when multiple persons are correcting a bug?

### Bug duration

If a bug is present for a longer time, does it affect our study?

## References

[The seven habits of highly effective GitHubbers](http://ben.balter.com/2016/09/13/seven-habits-of-highly-effective-githubbers/) by Ben Balter \(September 13, 2016\)

[A Data Set for Social Diversity Studies of GitHub Teams by Bogdan Vasilescu, Alexander Serebrenik, Vladimir Filkov](http://dl.acm.org/citation.cfm?id=2820518.2820601)

Hassan AE \(2009\)[Predicting faults using the complexity of code changes](https://www.researchgate.net/publication/221554415_Predicting_faults_using_the_complexity_of_code_changes). Proc. - Int. Conf. Softw. Eng. pp 78–88

Foucault M, Palyart M, Blanc X, Murphy GC, Falleri J-R \(2015\)[Impact of Developer Turnover on Quality in Open-source Software.](http://www.cs.ubc.ca/%7Empalyart/paper/2015_FSE_Impact_Turnover_Quality.pdf)Proc. 2015 10th Jt. Meet. Found. Softw. Eng. ACM, New York, NY, USA, pp 829–841

[A Unified Framework for the Comprehension of Software’s Time Dimension](https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/11998/Benomar_Omar_2015_these.pdf?sequence=2&isAllowed=y)

[14 Ways to Contribute to Open Source without Being a Programming Genius or a Rock Star](http://blog.smartbear.com/programming/14-ways-to-contribute-to-open-source-without-being-a-programming-genius-or-a-rock-star/) by Andy Lester \(March 10, 2012\)

## Authors

Simon Paris

Loïc Potages

Pascal Tung

