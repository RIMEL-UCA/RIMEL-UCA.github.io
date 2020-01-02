# **How to create the perfect team considering the actual code base?**

# Why ?

In big open-source projects, there is a mix of external contributors and a core internal time both contributing. Managing them is a complex problem and we would like to propose a solution in the form of a proposed team that suits the project’s architecture.

For that we want to use the opposite of Conway’s law that says that an organisation will build a system to its image by building an organisation that resembles the system it is working on.

Thanks to that, the project team will have a better internal structure permitting more efficient communication and enabling more productive work.

> @mbf : ok... j'espère que vous montrerez que la règle est vraie... et que vous la préciserez car "to produce designs"  et "communication structures" sont flous \(granularité,...\).

## How?

In order to determine the perfect teams, we need to extract data from the code base and the version control system.

First, we need to extract the code from the public repository. A simple “git clone” will do the job. As soon we have the code base and all the commit history, we can start to work on retrieving interesting data.

Next, extracting components from the code permits to slice the code base and use them to create a first team structure : a team composed by plenty of small groups that works only on a component.

> @mbf : ??? one component?

Sometimes, multiple components are strongly linked together, so retrieving them by checking which are the most often modified at the same time can form our submodules. We now have small groups working on submodules which we could call teams.

> @mbf : donc vous retrouvez des sous-systemes.. pas les équipes à cette étape.

Then, by analysing the history of commit to find links between committers and components we can start to populate our teams with contributors.

> @mbf : ok. Définissez un modele team, file, component, etc. Vous serez plus clairs.

Finally, to establish a hierarchy between contributors, we’ll retrieve the biggest contributors for each components this permits us to find who has the most knowledge of each part of the system and find appropriate technical leaders for each. Something identical could be used for modules in order to have the bigger picture.

Thanks to these analysis, we can derive a team that will have the good properties of having less external communication \(we group the people that need to work together\) and sharing the most knowledge of the same component \(all the people working on a component know the code and they are less likely to add a bug due to a lack of knowledge of the current code base\).

> @mbf : effective team et ideal team should distinguished.

In order to achieve what we have talked about we need some tooling. We’ve found those tools that permits us to do such things :

* code-maat,  to analyse the data from a version control system: [https://github.com/adamtornhill/code-maat](https://github.com/adamtornhill/code-maat)
* fractal figures, to visualize data: [https://github.com/adamtornhill/FractalFigures](https://github.com/adamtornhill/FractalFigures)
* neo4j to visualize, analyze and edit graphes: [https://neo4j.com/](https://neo4j.com/)

> @mbf : OK

## References:

* N. Nagappan, B. Murphy, and V. Basili. [The Influence of Organizational Structure on Software Quality](https://www.cs.umd.edu/~basili/publications/proceedings/P125.pdf). International Conference on Software Engineering, Proceedings.

  > @mbf : ok, TB

* M. D’Ambros, M. Lanza, and H Gall. Fractal Figures: [Visualizing Development Effort for CVS Entities](http://www.inf.usi.ch/faculty/lanza/Downloads/DAmb05b.pdf). Visualizing Software for Understanding and Analysis, 2005. VISSOFT 2005. 3rd IEEE International Workshop on.

  > @mbf : ok

* Conway, Melvin E. \(April 1968\), "[How do Committees Invent?](http://www.melconway.com/Home/Committees_Paper.html)", Datamation, 14 \(5\): 28–31, retrieved 2015-04-10

  > @mbf : ok

## Source code

For our study, we take as an example the Spring framework source : [https://github.com/spring-projects/spring-framework](https://github.com/spring-projects/spring-framework). We choose it because there is plenty of commit \(13,764 commits on the master branch the 19 december 2016\) and with 187 contributors we have a large amount of data to analyze.

> @mbf : ok

We want to use a project like it to be more accurate with the creation of the “perfect” team organization.

## Authors

* Tom DALL'AGNOL, SI5 \(5th year in Computer Science\) at the engineering school Polytech Nice-Sophia.
* Fabien VICENTE, SI5 \(5th year in Computer Science\) at the engineering school Polytech Nice-Sophia. 



