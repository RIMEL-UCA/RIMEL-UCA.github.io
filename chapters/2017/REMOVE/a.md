# RESULTS OF OUR TWO TEAMS

With our results we are now able to discuss about veracity of Conway’s law by combining the two researches done by each of our team. First of all, what do we expect to verify our promise?

According to our previous results, by merging what we findwe would have to find a team that has all the majors contributors who are in the same time zone and have the same leader for each submodule of Spring Project.

![](https://lh4.googleusercontent.com/3cBGBFWVeIXXj2wdJk4M0pmflbqCQUTv8SgIWid39GcDy1yw2n3eIKYN-rIH9nPLLZxdLkwTaxJV4UI3Ks08TMTTHkqkXOLJ_Co-HEfkp4zP_Wu4dnDt_iNNjNhj1MAk4E7bqFIU)

After collecting major contributors and associating with the results of the real team, all major contributors except Arjen Poutsma are committers and Juergen Hoeller which is the Project lead. In the previous results \(Tom and Fabien’s research\) they found that the contributor which has the major number of file owners is Juergen Hoeller and it results to be the project lead ! That’s a good point.

Arjen Poutsma was at the beginning a Committer and then he became a technical adviser in the same time zone that the project lead. Then we can see that for Spring Framework, most major committers are in the same time zone and depend on the same leader.

However, there are two people who are in completely independent time zones, so we have to look further to find out whether they are aberrant values or not.

First of all Phil Web is really interesting, because for all the 18 developers working on Spring Framework \(what’s Dorian and Manuel’s found\), he is the only one who is on UTC-8. What we found is that at the beginning of his career, Phil Webb was on the UK. He is a Spring Framework committer and Spring Boot co-founder and when he started to work for Pivotal \(the Spring company\), he started at the London agency \(so in UTC+0\) with Juergen Hoeller ! Now he changed to integrate Spring Web project so he decided to go at San Francisco, California to be closer to his team members, but he still help on Spring Framework project.

For Rossen Staoyanchev we have not found any specific information about his role and history at Pivotal, but we know he is a big committer in the project. He was initially an external commiter and integrated Pivotal after a few years but in the offices in Jersey City. He is the American referent of the teams based in europe but it is not in the same time zone …

# CONCLUSION

Firstly we can say that the projects we had access to were only open sources projects: the problem is that this kind of project does not have a well-defined hierarchy. Everyone can participate to the project code and most people works at home by modifying a little part of a component so we need to filter developers to find .

By looking at the repository, for sure we can get a lot of information, but it isn’t by evaluating the number of committed lines that we will really know if the person is currently participating to the project : somebody can have changed a lot of things \(by importing a library maybe\) without adding business value to the project unlike someone who has added 1% of the component lines and adding a big feature. In conclusion that metric is a good one but it is also misleading, maybe it has to be coupled with another filter \(maybe only the number of lines added ?\).

A good point is that with each of our two approaches \(from real team to hierarchy and from code to hierarchy\) we found that we had the same informations and by grouping our two results we found the same people who participated in the Spring Framework project.

Unfortunately the results we found do not allow us to validate if the Conway’s law is right or wrong, but we can affirm that for the Spring project it is not true because of the core team and note because of the external contributors \(the members of the teams we found are all internal contributors but they live on different time zone\).

Moreover Conway's law is a postulate made in the eighties, at that time there weren’t Agile techniques, versioning repositories and code sharing services \(git\) as we now have. The communication was not so simple as now and coding was much more complex with distance: we had real constraints of proximity and reachability.

The law isn’t verified for Spring project but we assume private companies that develop not open sources project are more attracted by Conway’s law because they need proximity on teams and developers.

# NEXT?

Our study focus on Spring that is made of multiple project, so we can’t conclude on all the open sources projects. We propose a methodology to apply our work on any open source project, with the only constraints is to have informations about the real team of the project. A plan can be to make an empirical study on a large amount of open source projects to see if we have only similarities between the real team and the code knowledge one, so that we can conclude either the Conway’s Law is true or false.

Another plan can be to prospect each real team, asking the enterprises directly. A problem that can emerged is that organizations don’t like to tell about their internal structure, so it can be pretty difficult to retrieve these data.

