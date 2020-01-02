# **How to create the perfect team considering the actual code base?**

## Methodology

The first thing we did is to try to model how we will represent teams and how it will map to concepts of a project. This model shown in figure 1 is the result of our work. It gives us a clearer idea on how to work on our project.

![](https://lh3.googleusercontent.com/T5ZiY2ZlKnxPBCxyzxpJVNkHuU_LSK6oV8Mu877hMaDuh0MnvedytXJmHCloREzoXHOVwhadFT5dHabDJux18S4JieSt716oGcO0Ki_sbUYnCV3cXvVxqRQNkPOf9JF8mTrYCkQr)

_Fig 1. Model of all the concepts we need for our research_

We have defined a couple of useful concepts. Our goal is to find the ideal team considering the source code. An ideal team is the union of all main contributors of a component, ie a team with all the people that have the most knowledge about the code. We define a component as the one gave by the spring-framework : all folders at the root containing a source folder. A file is a source file \(.java in the spring case\). A contributor of a file is someone who worked on a specific file \(had commit link to that file\). An owner is the contributor which has at least 50% of the code of a file. From these, we find an efficient team by component : the gathering of all owners of one file in the component. At the end, we have a team by component.

We first use code-maat to extract data from the Spring repository. It retrieves all the commits and allows us to know which contributor has the most ownership on a file. To define a component which will be useful later, we take advantage of our definition of a component: each folder container a src folder. We use a python script to scan each component and file that are linked together. We also link components to the project. Then we export the ownership data we previously obtained into the database. This script will use the component data which is already present at this point to link owners to their respective component by searching to which component pertain the files they own. At this point we can dispose of file data and we will have a project, with components and developers that are linked together that can form a core team. But this isn’t exploitable yet because of the sheer quantity of data we would prefer something simpler. To this goal we devised a heuristic. Simply put, the more a contributor participates in components the more he should understand the whole project and, purely technically speaking he should be higher in the hierarchy. Using this idea we wrote a query in Neo4J’s language cypher. This query gives us a score \(the number of components someone has ownership in the project\) that we will use to try to form our team.

Tools:

* code-maat : a tool to analyze a version control system repository \(as a git repository\) and extract information as the ownership of files, the coupling, or the effort.

* python : to input the code-maat data and data analyzed into neo4j. We use it to transform the raw data from code-maat into something understandable by neo4j. We choose it because we have experience with this language, so it’s quick and easy to transform data. Another advantage is that there is an API to link python directly to neo4j.

* neo4j : A graph database, to put the data retrieved by both code-maat and python and then use request on it to extract the meaningful representation of our data \(e.g. : the team members that shared the most knowledge on a project\). A graph database is the tool we needed because it permits to work on a heavy data set that have linked to each other, and make request on it to build a specific model. In our case, the authors are linked to components by ownership, and these are linked to projects.

Expectations:

We expect to find the same team than the one extract from the other team \(the real team\) because they are paid to develop : they should have the most ownership from all the contributors \(they produced more code because they spent their workdays on the project\). We also expect to always have someone that stands out, the one that work on each component to link them to each other, the one that have the most hindsight on a project : the team leader.

## References

* N. Nagappan, B. Murphy, and V. Basili. [The Influence of Organizational Structure on Software Quality](https://www.cs.umd.edu/~basili/publications/proceedings/P125.pdf). International Conference on Software Engineering, Proceedings.

* M. D’Ambros, M. Lanza, and H Gall. Fractal Figures: [Visualizing Development Effort for CVS Entities. Visualizing Software for Understanding and Analysis](http://www.inf.usi.ch/faculty/lanza/Downloads/DAmb05b.pdf), 2005. VISSOFT 2005. 3rd IEEE International Workshop on.

* Conway, Melvin E. \(April 1968\), ["How do Committees Invent?"](http://www.melconway.com/Home/Committees_Paper.html), [Datamation](https://en.wikipedia.org/wiki/Datamation), 14 \(5\): 28–31, retrieved 2015-04-10

## Results

Using those methodologies we extracted data from nine repositories which are part of the Spring organisations. These project aren’t chosen at random but are the one for which the other contributing team have the most data on. First of all, here are some numbers about the raw data we have extracted:

* 20000+ files analyzed

* 322 contributors

* 65 components

* 9 repositories

Here are some captures of what the team graph linked to component looks like

![](https://lh4.googleusercontent.com/qJphyZtNzGPipjY-0zc18lxnzcohtXZAxzxvrykG4kfmvpuvw_dpXVBZlljyN4xo3QPR5KWGuTNCPxk3MuuooGPvljD5lanu4adSV77mqk8Or45LqmYFiLnLotrSLn879RqevW_b "graph-spring-framework.png")

**Graph of spring-framework**

We can see that on spring-framework we have a lot of component and a set of same main contributors to each of it.

![](https://lh6.googleusercontent.com/BEuw9hjnZGykRRFEPzSuyRsf0jzGGLQkuNqGNJMo960u-dVzGNP-33NFl-vke96-7mC0EZmjKvxRrzWpCqpNU_rKMxdzILAX0vMWCkpMhMQe03tNzG2eXAIINN0_nFb_hiM-7A1T "graph-spring-batch.png")

**Graph of spring-batch**

Here, we have fewer components \(8\) but a large amount of main contributors \(26 persons\)

![](https://lh4.googleusercontent.com/E6Y4auUjgnDqapz02DMsWPjhboQB0VgokErJ-msrsu0c_sYBFv_Zem8ItLoZd0RMGtAO33szmoZUNkI3YLDqLtUyORfLUiQZrcsQKEILL9cOo7RmI_O8x4T-OZyOUl-K2vozbTvR "graph-spring-retry.png")

**Graph of spring-retry**

Spring-retry contains only a few components \(3\) and only 12 contributors. It’s really few comparing to the two previous graph.

From there we focused our efforts on spring-framework which, being the biggest project of Spring’s team, was the one with the most data available. So we devised a heuristic to make sense of it. This heuristic, as explained in the methodology part gives us a score on which we can base ourselves to create a team.

![](https://lh6.googleusercontent.com/vuYPJxFZQssQgVQaOn9nmK1U9y-B4CXXv2qYSqXRbiCrZ7iJ-AdSAwYEPt3VU87jdhiiRNXE_R-8tOvrvSCjYlo5uVx6DJMiqso6vKPo4SbBmc1fe7iuFQ1IwbxtKmInizB_6BYw "ownership number.png")

From there on we had all our required data to form a core team, meaning a small team of people which all have the core competencies. The choice is not innocent, our goal is to be able to compare our teams and the other one can only retrieve the core group, so this is the one we are looking for.

![](https://lh3.googleusercontent.com/ZELgVLUsVGHuxUrNFsc_blwYyUDt38r1KgrY4znm15a0VvAVm97Gz4xj92Ayc2Y_XZeh3MkhMLLAw5wo_0331_FZDSc6wyrp4qV_SQHtG1FXbssw_hibUVO-1tx0kQGjl7rhFwWv "team-code-knowledge.png")

Using our extracted data we could form a team with only 7 member. The hierarchy here is only a example but we could see that Juergen Hoeller emerged as a leader. Now that we have this data, it is time to merge both teams and see if we can conclude our research.

