---
layout: default
title: Study of the relation between algorithms, preprocessing and their usage on time series
date: 2022-01-10 22:00:00 +0100
---

**_janvier 2022_**

## Authors

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

- Eric Boudin <eric.boudin@etu.univ-cotedazur.fr>
- Clément Monestier <clement.monestier@etu.univ-cotedazur.fr>
- Florian Naud <florian.naud@etu.univ-cotedazur.fr>
- Lucas Rakotomalala <lucas.rakotomalala@etu.univ-cotedazur.fr>
- Loïc Rizzo <loic.rizzo@etu.univ-cotedazur.fr>

## I. Research context /Project

OpenML is an online, collaborative environment for machine learning where researchers and practitioners can share datasets, workflows, and experiments. It is particularly used for meta-learning research; by studying a large number of past experiments, it should be possible to learn the relationship between data and algorithm behavior[1].

In this project, we want to extract knowledge about time series processing workflows.

_Study of the relation between algorithms, preprocessing and their usage on time series_

We want to know the links that may exist between data and
how they are handled. We could find a relation between
these which would lead us to better understand the best context of use
algorithms.

## II. Observations/General question

A. What studies are performed on time series? What are the major types of tasks
on these data in OpenML (Classification, clustering, anomaly detection)?  
B. What are the most frequently used algorithms and preprocessing? In what
order are these algorithms called? Can we identify sub-workflows, joint
occurrences of the same algorithms?  
C. By analyzing the experiments, can you identify preconditions on the
algorithms? Are there any algorithms that are always present?  
D. Are there any algorithms that are only used on time series?  
E. What population is working on the time series? Is there an evolution over time?

## III. information gathering

We can extract the completed tasks and flows thanks to the OpenML API.
By using an existing library ([OpenML](https://pypi.org/project/openml/)) and with our code in Python.

## IV. Hypothesis & Experiences

We think that it exists some alogrithms and preprocessing that are better used in some conditions than others.  
Our focus will be to find the different approach used by searchers on OpenM.  
Then study the link between data, algorithms, preprocessing, flow and result.  
Finaly visualize our results.

## VI. References

[1] V. J. N. Rijn and J. Vanschoren, “Sharing RapidMiner Workflows and Experiments with
OpenML.,” in ceur-ws.org MetaSel@ PKDD/ECML, 2015, pp. 93--103, Accessed: Dec. 17, 2021. [Online]. Available:
ftp://ceur-ws.org/pub/publications/CEUR-WS/Vol-1455.zip#page=98.
