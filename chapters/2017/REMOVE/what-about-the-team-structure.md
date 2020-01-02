# What about the team structure ?

### _Intro:_

This study tries to compare the a  model of an ideal team, based on Conway’s law, to a real company team. In this article we will expose some criterias of the Spring.io team structure.

First of all, we need to extract the team from the website project. To achieve this, we choose to develop our **own **extractor, in few word: given a webpage \(one extractor is write for one page\), we can by **parsing **the tags and make analysis catch the information needed.

Extract tool is available at [RIMEL-extractor](http://rimel.dobl.fr/) \([http://rimel.dobl.fr/\](http://rimel.dobl.fr/\)\)

With the extractor tool you can:

* Choose a project which reference a team \(e.g. [https://spring.io/team\](https://spring.io/team\)\)
* See the list of team members
* The Map will shows you their location
* The grid shows you their team position \(e.g. Developer\)
* Sort by Name, Position, Location, Github name
* View the most represented positions and locations
* View the correlation between project and location \(Coming soon\)

We can now analyse the team by considering some metrics: the projects that Spring team develop, the projects locations and the team positionning.

We need to make sense of the data we have collected.

First, the data we can collect on the spring.io/teams page is not extended and knowing that they have certainly been filled by hand, they are not very reliable.

So, we have begun to extract the geographical position of the developer, his main position, his github adress and his name.

### 

### Project_: \(comming soon\)_

We will search in the repository of each developer looking for projects which he works at spring.

And we will pool all the members who work on this project whose geographical position and the name of the team are known and determinate if the team is too separate or not.

