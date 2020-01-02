# How can user requests impact modular architectures ?

## Introduction

In this chapter, it will be discussed "_how can a community drive the developpement of a software \(in our case a video game and 3d engine\) or at least some features? _". This can be achieved by the development team, or driven by the community playing the video game \(or using the software\).

_**Questions :**_

* How can a community impact the development of some features ?
* How long does it take from the idea to the first realisation ?

With these questions, we hope to retrieve and measure some information like :

* the mean time between a demand and a commit
* the number of users implied in the community 

_**Projects to study :**_

* [Terasology](https://github.com/MovingBlocks/Terasology) : Terasology is, in the first place, a clone of the well known game Minecraft. Now it is more like a 3D engine that you can use to make your own game.

_**Methodology :**_

* Analyse the community.
* Analyse the Git repository.
* Attempt to find every pull request involving change in plugin interface system.

---

## Analysis

_**Subject:**_

After a few weeks of studies, we think about reducing the scope of this part, because there are too little documentation on it. We will focus on how the community drive the development.

_**First Methodology:**_

We will run through 3 phases to analyse data from the Terasology repository \(for instance\).  
The first phase will be to extract data : to do this, we use an homemade script and we extract :

* The commit hash
* The author name
* The author email
* The author date
* The committer name
* The committer email
* The committer date
* The subject 

The second phase is to transform data using Logstash to convert the CSV, product by the fisrt step and produce a JSON file. We then use this JSON to populate an Elasticsearch.

The third and last step is to exploit these data, and produce charts, stats, dashboards, ... and reason on it.

_**First results:**_

We were able to extract,  display the files per commits over the last two years. We can measure that over 1300 modifications was made between November 30th and the December 6th.

![](/assets/screen_kibana.png)

_Figure 1 : Our first result in Kibana._

---

_**New Methodology:**_

An idea suggested by one of our teachers  was to going back up to the source of our community : their forum. We then implemented a crawler in order to deduce relevant information \(or even piece of information\) like :

* Who is the most talkative on the forum
* Who talks about features 
* The date when people asks for features
* The elapsed time between the day a user makes a request and the day of the first commit about this request

Here are all data we retrieved from the forum :

* Title of the message
* Pseudo of the writer
* Date of the message
* Category of the message
* Content of the message
* Content of the message without useless words \(determinants, subjects, ...\)
* URI of the page

The crawler generates data ready to be insert in ElasticSearch.

We also extracted information about pull requests and issues on Git using the Githup API, converted them into proper JSON object and placed them into our ElasticSearch.

To summarize, our elasticsearch now contains :

* All github commits
* All github issues
* All github pull requests
* All forum messages

_**New results:**_

We first extracted the messages from the forum in order to have an overview of distribution of the messages among all the users. It appears that the person named 'Cervator' produces or answering to most of messages of the forum. We found out that this person spearhead the Terasology project. Immortius is also well known by the community because he answers to a lots of demands, and lead some development.

![](/assets/community_1.png)

_Figure 2 : The coarse grain distribution of users in the forum._

We decided to dig among the 6.4% other person on the forum to see if the distribution of the messages becomes homogeneous.

![](/assets/community_2.png)

_Figure 3 : A more fined grain distribution of users in the forum._

Here we can see that the distribution is a little bit more homogeneous. These people  \(up to 4%\) are quite active in the Terasology community.

We then tried to know what were the principle discussions on the forum, by taking a look at this wordcloud \(containing some of the main subject's title\) :

![](/assets/words.png)

_Figure 4 : Main words used as title in the forum._

We then decided to take a look at the git contributor in order to find an overlap with the people on the forum. And came out withh this graph :

![](/assets/contributors.png)

_Figure 5 : The distribution of contributors._

We were facing a new problem : naming. While some person give their name both on Git and on the forum, most of the users have different name on both plateform. For instance "Rasmus Praestholm" is the "Cervator" named above. Here again, we had no automatic way to link the person on the forum with the person on Git. Therefore we did it manually. "Immortius" named above is here known as "Martin Steiger" for instance. And so on...

Then, we decided to take a look a the commits to see if we could find some commit name linked to the wordcloud just above. We couldn't find a way of doing it automatically because of the gap of the two languages. We did it manually and tried to have as much as connections as possible. We came out with 10 related subjects between the repository and the forum. We were then able to compute some measures like the time between the message in the forum and the commit. We came out with theses values :

* Min : - 5 hours \*
* Mean : 5 months
* Max : 1 year

_\( \* : The person acutally posts in the forum to say "I did it, do you like it ?" AFTER committing on  the repository.\)_

---

## Conclusion

We came to the conclusion that this project was poorly driven by the community. Only a bunch of people maintaining up to date, the people on the forum rarely propose new ideas. It is more like the leaders say "We suggest THIS, do you agree?". The link between the information retrieved from the forum and those retrieved from Git was tedious and not really automatisable which prevent us from gathering enough data to our research.

_**Major difficulties:**_

We encountered major difficulties in term of documentation. Searching tools has also be a pain because there are too few researches about it.  
Language has also be complicated. While in the forum everybody use a natural language \(English most of the time\), the issues and the pull requests reference to the code itself so it was impossible to find an automatic way to match the natural language and the code.

_**Tools used:**_

We first thouugh of using Diggit but we realized that this tools wasn't fullfilling our needs : we don't know Ruby, and the possibilities are too limitied for our purpose. So we use homemade scripts and the ELK stack where we mainly use Elasticsearch  and Kibana. We also implemented our own crawler for the forum.

* [Github API](https://www.gitbook.com/book/mireillebf/uca-students-on-software-maintenance/edit#)

* Scripts

* Homemade Crawler

* ELK

_**References used:**_

* Software modularity :  [Investigating software modularity using class and module level metrics M English, J Buckley, JJ Collins - Software Quality Assurance: In …, 2015 - books.google.com](https://www.gitbook.com/book/mireillebf/uca-students-on-software-maintenance/edit#)

_**Distribution of Work:**_

* MANNOCCI Adrien \(M2\)
* SARROCHE Nicolas \(SI5\)



