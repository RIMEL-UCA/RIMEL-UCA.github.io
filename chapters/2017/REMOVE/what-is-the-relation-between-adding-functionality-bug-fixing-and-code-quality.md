# _What is the relation between adding functionality, bug fixing, and code quality ?_

## _**authors**_

BELHASSEN Issam

DENNE Djoe

DESTORS Max

MOULAYEELY Bezeid

## _Abstract_

In this question we will try to highlight any relation between adding functionality, bug fixing, and degradation or improvement of code quality. We hope to highlight an overall trend in the quality of the code over time.

The idea is that we take several criteria on the quality of the code that we are going to determine and we look at the evolution over time.We are looking for a correlation between the addition of functionalities or the correction of bugs and the quality of the code and looks at several curves if the evolution of the quality corresponds or not to the correction of bugs or the addition of functionalities, etc.

## _Problematic_

The importance of our question comes from the relation between the code changes and its quality,we are trying to verify the rumors that say there  is a perception that when new features are added to a system that those added and modified parts of the  source-code are more fault prone. Many have argued that new code and new features are defect prone due to immaturity, lack of testing, as well unstable requirements also we find that frequently changed code is often buggy.

But are these beliefs well-founded?Is there evidence to support the belief that bug fixes, additions, and feature enhancements can increase or abbreviate code quality?

## _Context and hypotheses_

### _Context_

It is always annoying to see the progress of a project slow down, either because of the frequent occurrence of bugs or because of the degradation of code quality, which makes development much more complicated.

There are a number of tools to solve the problem: Sonar, Code Maat ... Of course, the use of its solutions allows to guarantee a clean and easily maintainable code or to progress towards this state. But none of these tools can extract value from this data. There is no additional information available for this project. A project is available in this kind of situation: Is it due to a lack of skills of the team? The context of the project that prevents the code review? Are there times when code degradation is more important? Can some habits or processes affect or improve the quality of the code? Or is it due to the addition of functionality that burdens the code or the bug fix done precipitously?

It is precisely this question that we have chosen to answer. While we know that if we can set up a process to answer this question, the effort to answer the other questions will greatly simplify. This is proof of concept and feasibility.

To do this we chose to base ourselves on the Scala project which is an open source project available on Github with a public ticket manager under Jira. It has a lot of the information we need but like all Open Source projects it suffers from a lack of compliance and rule, especially in commit messages. We will detail are point later.

### _Hypotheses_

To answer this question we need a project that gives us access to a Git repository and to its ticket manager. In our case, we opted for an open source project under Github and Jira. But the process described and valid for all projects with a Git directory and a ticket manager. The tools and scripts developed are based on the Github and Jira APIs.

Our biggest constraint is at the level of the commit messages, which must be formatted in order to integrate the Jira ticket number or the Pull Request number to which it responds.

In absolute terms, a strong link created for example by a plugin allowing to link the deposit Git to the Jira would be a not negligible and would make it easier to extend the subject to other question.

To be certain that the process described here works effortlessly to provide, it is necessary to:

* A Github depot.
* A Jira ticket manager. Tickets must be linked to versions \(Affected version\)
* The ticket type must be specified. They should not all be "task"
* Tickets describing a Pull Requests must have the tag "has-pull-request" and have the Pull Request link in a comment.
* Commits must include a ticket or Pull Request number in their message.

## _Target project_

Let's make a statement of what we have. So we have a Project, Scala, which has a Github with a core of contributors and external contributors. The project also has a Jira on which you will find all the tasks related to a version. Tickets can be of 3 types only:

* Bug.
* New feature.
* Improvement.

It is managed by version, the tickets being linked to a specific version, and on the Git finds a branch by minor version.

External contributors are required to create Pull Requests to contribute to the project. Accepted and merged Pull requests have a Jira ticket associated. Tickets created for Pull Requests are tagged "has-pull-request" and the Pull request link is commented.

There are currently 26,000 commits and we initially chose to focus on versions ranging from 2.10.x to 2.12.x, or nearly 3600 Jira tasks.

## _Process_

In order to answer this question we have put in place a process which allows to extract different metrics which will allow to put forward any correlation between the different nature of modification and the evolution of the quality of the code.

We will first have to make the link between the commits and the tickets to which they respond, which allows us to have a link between the commit \(and thus the code\) and a nature of modification. Then for each commit a differential analysis between this commit and the commit directly preceding it will be launched with Sonar. This will give us a collection of differential analysis for each nature of modifications, then we can get out of the value by doing various statistical processing \(average, standard deviation, variance ...\).

### _Link between Code and Jira_

The first step in our process will be to link the commits to the 3600 tickets we are interested in. The information we will need for the rest of the process is the list of commits for each type of ticket and for each version. A commit will be represented only by its SHA and the previous commit's \("parent"\) SHA which is the information to retrieve the commit from Git for analysis.

To get this list we will first have to list the commits and extract their key. The key is the information that allows to link this commit to a ticket, that is to say are either "ticket number" or its Pull request number if there is one. The 26 000 commits are therefore retrieved by 100 per Github API and their message is analyzed to extract the key. In parallel to this treatment, they are grouped by their keys. So we get a list of keys with a list of SHA peers, each of which represents a commit, and are commit "parent".

We will then have to extract the tickets from the versions we have chosen to analyze. These tickets will be represented by their key and are type. So if a ticket is tagged "has-pull-request" it will retrieve all its comments to analyze them and find the url of a Pull request, this one will contain the number that interests us. If a ticket is tagged "has-pull-request" but we are unable to retrieve its number, the ticket will be ignored. Once the key is extracted we will try to find the commits linked to this ticket thanks to the list of commits previously generated. Tickets are also grouped by type, so either bug, feature addition or Improvement.

So we have at this moment a list of versions with each having three list of tickets represented by their key and a list of peers of commits, they are grouped by type.

We gathered and coupled the information necessary to be able to analyze the code by the Sonar tool and to correlate the results with the information already obtained, that is to say the types of ticket.

### _Launch sonar analysis_

With the output of the first script, we compute a list of Pull Requests with some data. At this stage a PR do have a key, which is an identifier, a version, the linked version of the project, a type, the type of modification given by the linked ticket and the two SHA of the linked commits. The Parent commit, which is the commit just before the merge of the pull request, and the Merge commit that apply all the modifications done by the pull request.

The script pass trough this list to setup Sonar properties and launch a sonar scanner. To make the gather of the Sonar data easier we decide to proccess as follow. For each Pull Request, we run two Sonar scanners, one for each commit. At the Parent commit we create a new sonar project with version number 1 and at the Merge commit we only change the version number. This way we can later retreive the number of issues closed and opened using the Since Leak Period. To acheive this behaviour the script, for the Parent commit, change the content of the 'sonar-project.properties' file on keys projectKey, projectName and projectVersion using the PR key identifer. Then using a 'git checkout SHA' we place the git repository at the Parent commit and we run a Sonar Scanner. For the Merge commit we do the same but we only change the version in the sonar properties file.

A scanner of the projet takes about 1 minute and a half but the SonarQube must then Analyze the scanner and this takes about 4 minutes. This second analysis can be done in background but unfortunatly it happen to have conflicts between the analisys. So we decide to wait a bit between each Scanners.

At the end of this script we write in a file the PRs that have been scanned in a json file used by the next Script.

### _Gather Sonar data using the web API_

In this script we first get the PRs scanned using the json file of the previous script. Then we do many calls to the SonarQube web API to gather number of new Sonar Issues, the number of Closed Sonar issues and the current total number of Sonar Issues \(after PR modification\).

Those three calls are done 5 times, one for each severity level \(INFO, MINOR, MAJOR, CRITICAL, BLOCKER\).

All this data is stored in a json output and a csv output. The json contains all the data gathered through all scripts. And the Csv only have the data we will use for graphic and data interpretation. Which is, for each line, the PR key identifier, the type of modification, the verion of the project, the severity of sonar issues, the number added Sonar issues, the number of removed sonar issues, and the current total number of issues.

## _Problems_

We encountered many problems in the process. One of the first was the lack of consistency in commit messages. This problem is probably related to the open source aspect of the project but it had a significant impact on the results obtained, in the end among the 3600 selected tickets we succeeded in finding the commit link in only 40% of the cases, which can Greatly distort the results and amputates us from much of our test data.

The second problem is related to the Sonar analysis of the commit. Our process, although functional and complete, lacks optimization. So for each commit we run a full scan of the source code instead of running it only on files impacted by the commit. This rendered the extraction of results extremely complicated, an analysis may take several minutes.

And finally, we realized a little late, that the Scala project benefited from a regular Sonar analysis. This introduces a variable whose effects it is impossible to predict on our test set.

## _Results_

Unfortunately, we have no interesting results. The problem posed is not in question, but rather the slowness of the process and the project chosen.

The project seemed to match our criteria: Open Source, with a ticket manager and the commit messages seemed in many cases to have information to link it to a ticket. Although the Pull Request mergers have all this information, it is not necessarily the case of the other commits. This aspect is linked to its nature Open Source which makes it necessarily less constant in compliance with certain rules. To this is added the fact that it regularly ran a Sonar analysis, which distorted the few results that we were able to extract.

The process is also involved, we not in its run but in its current execution time which is about 6-7 minutes for each couple of commit. It is a correctible defect, but unfortunately we did not have time.

## _Evolution_

We realized that the process we put in place could answer a lot of other questions with relatively few changes.

Indeed by modifying the selection criteria of the commits which is today the different types of modification and returning more information related to the commits when the first phase, information related to the context \(date, time compared to a release ... \) Or the developer, we could derive much more value from it:

* Is there a link between the evolution of the quality of the code and the period of the year where we are?
* Is there a link between the evolution of the quality of the code and the reconciliation of a deadline \(date of release\)?
* Do Pull requests from external contributor tend to further degrade the quality of the code?
* Who codes the most cleanly?
* Who corrects the most Bad smelt?
* Who degrades the quality of the code?

And we could answer all these issues relatively simply by smaller modification of the process.

## _Tools used_

-github

-Jira

-sonar

### _Sonar_

In order to be able to evaluate a code we will analyze it to know the quality and evolution in time of the code. For this we ill use Sonar.

Sonar is an open source tool that supports the development and support of Sonar.

The main purpose of this tool is to provide a complete analysis of the quality of an application by providing numerous statistics \(or metrics\).

These data allow to evaluate the quality of the code, and to know the evolution during the development.

The sonar tool will allow us to :

* Detecting a smell code
* A quantitative measure of the class number and duplicate code
* For projects where there are unit tests, sonar allows us to have a qualitative measure of coverage and success rate of tests
* A history to see the evolution over time

## _References_

\[0\] Meir M. Lehman - Programs, Life Cycles, and Laws of Software Evolution - [http://www.ifi.uzh.ch/seal/teaching/courses/archive/FS13/SWEvo13/lehman-IEEE-80.pdf](http://www.ifi.uzh.ch/seal/teaching/courses/archive/FS13/SWEvo13/lehman-IEEE-80.pdf)  - 1980

\[1\] Lehman, M. M.; J. F. Ramil; P. D. Wernick; D. E. Perry; W. M. Turski - Metrics and laws of software evolution—the nineties view [http://users.ece.utexas.edu/~perry/work/papers/feast1.pdf](http://users.ece.utexas.edu/~perry/work/papers/feast1.pdf)  -  1997

The evolution of the laws of software evolution. A discussion based on a systematic literature review

[Herraiz I, Rodriguez D, Robles G, Gonzalez-Barahona JM \(2013\) The Evolution of the Laws of Software Evolution: A Discussion Based on a Systematic Literature Review. ACM Comput Surv 46:28:1--28:28.](http://www.cc.uah.es/drg/jif/2013HerraizRRG_CSUR.pdf)

## 



