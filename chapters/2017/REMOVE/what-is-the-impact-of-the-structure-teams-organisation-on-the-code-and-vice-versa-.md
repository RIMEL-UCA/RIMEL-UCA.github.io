# Is Conway's law valid for an open source project?

---

_**Team:**_

* Dorian BLANC
* Fabien VICENTE
* Manuel PAVONE
* Tom DALL'AGNOL

_**Tools: **_

* GitHub API
* TeamCodeParser \(Made by us, allows, given a Github repository \(ex: github.com/spring-io/sagan\), to extract the team by **code exploration**\)
* TeamParser \(Made by us, allows, given a webpage \(ex: spring.io/team\), to extract teams by **subject **and **location**\)

_**Project to study: **_

* [Spring](https://spring.io/)

_**Questions: **_

* Is Conway's law valid for an open source project ? 
  * Analyse the real team and try to match the ideal team according to conways law
  * Extract the code and analyze it to determine the perfect team.
  * Merge the results to be more accurate and verify the suatability of the law

## Introduction

In 1967, Melvin Conway introduced at the National Symposium on Modular Programming the idea that “Organizations which design systems \[...\] are constrained to produce designs which are copies of the communication structures of these organizations”. A rule that was true at this time, but with the propagation of open source philosophy, we want to know if it’s respected.

Open-source projects are a quite recent evolution in informatic with the propagation of free repository hosting like github \(launched the 10th April of 2008\). In big open-source projects, there is a mix of external contributors and a core internal team both contributing, that doesn’t know each other, something far from what could have imagined Conway at his era when there was only static teams.

We want to know if the current teams of open sources projects have the same organization than the one we could extract from the code. Looking at the code and the commits of a Version Control System \(VCS\), we could extract a design that, according to Conway, should be a copy of the real team.

If we find that the team created using the organizations informations and the team created using the code source is the same, we could conclude that Conway’s Law is valid.

We chose Spring because it’s a set of Open Source projects on which we have informations on the current teams. We can also link from their website the team members location and get their github account to have more informations. That’s why these projects are perfect for our study.

