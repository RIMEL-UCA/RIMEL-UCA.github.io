## How do contributions of developers evolve over time?

_**Project studied:**_

The open source projects are a good study case because there is often many move, developers start to contribute to the project and other stop to contribute. We should have many data to analyse the developers attitude.

We will study the [ElasticSearch](https://github.com/elastic) project.

_**Methodology:**_

We will start by getting Github data out to work on it. The Github API should make possible the extraction of these data. Depending on the data format we will use for example python for processing of data. Also with python we could make a first set of basic graph for guide the rest of our exploration.

We will use the command _git l_og to obtain the author, the date and the message of every commit of a project. With that we will try to define som developers profils : check his volume of commit and make a lexical analysis on each message of commit, for quickly determine if it is a update for documentation, test, feature... We could determine when the most of the developpers in a open souce project start to fix bug. If they start direcly by fix bugs or make some others works for learn more about the project before that.

Depending on the things that we could reveal with data, we will choose the good graph tool for representing them. Considering the huge quantity of data that we can extract, we could use the tools of the project that we will study : ElasticSearch, and Kibana to make dashboards.

The capability of ElasticSearch and kibana will surely be usefull to search specifics words into the commit message. If we could define strings associate to kind of commit \(ex: "fix bug" in commit message could be enough for consider that is a fix commit\) we will be able to make visualisations that will show the activities of developer and most particularly if there is any order in this activities.

_**References used:**_

[The seven habits of highly effective GitHubbers](http://ben.balter.com/2016/09/13/seven-habits-of-highly-effective-githubbers/) by Ben Balter \(September 13, 2016\)

[A Data Set for Social Diversity Studies of GitHub Teams by Bogdan Vasilescu, Alexander Serebrenik, Vladimir Filkov](http://dl.acm.org/citation.cfm?id=2820518.2820601)

_**Tools used:**_

Git commands

Github API

Python for manipulate data and make a first set of graph

ElasticSearch / Kibana



